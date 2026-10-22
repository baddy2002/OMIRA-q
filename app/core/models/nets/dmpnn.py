from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax


def find_reverse_edge_indices(edge_index: torch.Tensor) -> torch.Tensor:
    """Trova per ogni semiarco e=(u->v) l'indice del semiarco inverso e_rev=(v->u).
    Questo mapping è il cuore matematico della prevenzione del backtracking nel D-MPNN.
    """
    row, col = edge_index
    # Codifichiamo la coppia (u, v) in una singola chiave intera univoca
    num_nodes = max(row.max().item(), col.max().item()) + 1 if edge_index.numel() > 0 else 1
    edge_keys = row * num_nodes + col
    rev_keys = col * num_nodes + row

    # Usiamo un dizionario di ricerca su CPU/GPU tramite torch.searchsorted o hashtable
    # Per stabilità e velocità in PyTorch:
    sorted_keys, perm = torch.sort(edge_keys)
    rev_indices_in_sorted = torch.searchsorted(sorted_keys, rev_keys)
    
    # Gestione di eventuali edge senza corrispondente inverso
    clamp_indices = torch.clamp(rev_indices_in_sorted, 0, len(sorted_keys) - 1)
    match = (sorted_keys[clamp_indices] == rev_keys)
    
    rev_edge_indices = torch.where(match, perm[clamp_indices], torch.tensor(-1, device=edge_index.device))
    return rev_edge_indices


class DirectedMessagePassingLayer(MessagePassing):
    """Layer D-MPNN rigoroso: i messaggi risiedono sui semiarchi (u -> v).
    Il messaggio verso (u -> v) aggrega i messaggi dai semiarchi incidenti (w -> u)
    ESCLUDENDO tassativamente il semiarco inverso (v -> u) per eliminare il backtracking.
    """

    def __init__(self, hidden_dim: int, num_edge_features: int, heads: int = 4) -> None:
        # Aggregazione sui semiarchi
        super().__init__(aggr="add", flow="target_to_source")  
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim deve essere divisibile per heads"

        # Proiezione delle feature iniziali del legame e degli atomi
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attenzione multi-head sui semiarchi orientati
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Proiezione finale per ricomporre le teste
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_node: torch.Tensor,
        h_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rev_edge_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_node: [num_nodes, hidden_dim]
        h_edge: [num_edges, hidden_dim]
        edge_index: [2, num_edges] (row=source u, col=target v)
        rev_edge_indices: [num_edges] con l'indice dell'arco inverso v->u
        """
        row, col = edge_index  # u -> v

        # 1. Costruzione dei vettori dei semiarchi combinando nodi e legame
        edge_input = h_edge + h_node[row]
        num_edges = edge_index.size(1)
        num_nodes = h_node.size(0)

        # 2. Calcolo dei vettori Query, Key, Value per l'attenzione
        Q = self.q_proj(edge_input).view(num_edges, self.heads, self.head_dim)
        K = self.k_proj(edge_input).view(num_edges, self.heads, self.head_dim)
        V = self.v_proj(edge_input).view(num_edges, self.heads, self.head_dim)

        # aggregazione dei messaggi in arrivo a ciascun nodo target (col)
        # Sommiamo tutti i vettori V in arrivo al nodo col
        node_incoming_V = torch.zeros(num_nodes, self.heads, self.head_dim, device=V.device)
        # Usiamo index_add_ per accumulare i messaggi sul nodo destinazione (col)
        col_expanded = col.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.head_dim)
        node_incoming_V.scatter_add_(0, col_expanded, V)

        # mascheramento anti-backtracking
        # per ogni arco u -> v (indice i), il target è j (col[i]).
        # i messaggi in arrivo a j includono anche quello proveniente dall'arco inverso v -> u.
        # per escluderlo, prendiamo la somma totale in arrivo a j e sottraiamo il V dell'arco inverso.

        # isoliamo il V dell'arco inverso per ciascun arco
        valid_rev_mask = (rev_edge_indices >= 0)
        V_rev = torch.zeros_like(V)
        if valid_rev_mask.any():
            # dove esiste l'arco inverso valido, preleviamo il suo valore V
            safe_rev_indices = torch.clamp(rev_edge_indices, 0, num_edges - 1)
            V_rev = V[safe_rev_indices]
            # Azzeriamo dove non c'è inverso valido
            V_rev = V_rev * valid_rev_mask.unsqueeze(-1).unsqueeze(-1)

        # somma totale in arrivo al nodo target col - V dell'arco inverso)
        incoming_sum_at_target = node_incoming_V[col]  # [num_edges, heads, head_dim]
        filtered_incoming = incoming_sum_at_target - V_rev

        # calcolo dell'attenzione MHA tra Q (arco corrente) e filtrati K
        scores = (Q * filtered_incoming).sum(dim=-1) / (self.head_dim ** 0.5)  # [num_edges, heads]
        alpha = F.softmax(scores, dim=-1)  

        # messaggio finale ponderato per attenzione arco
        m_edge = filtered_incoming * alpha.unsqueeze(-1)  # [num_edges, heads, head_dim]
        m_edge = m_edge.reshape(num_edges, self.hidden_dim)
        m_edge = self.out_proj(m_edge)

        # aggiornamento delle feature di nodo e di arco
        node_agg = torch.zeros(num_nodes, self.hidden_dim, device=h_node.device)
        node_agg.index_add_(0, col, m_edge)

        node_input = torch.cat([h_node, node_agg], dim=-1)
        h_node_new = h_node + self.node_update_mlp(node_input)
        h_edge_new = h_edge + m_edge

        return h_node_new, h_edge_new

        # 3. Mascheramento anti-backtracking durante la propagazione
        # Calcoliamo la propagazione MHA orientata inoltrando i tensori
        m_edge = self.propagate(
            edge_index,
            Q=Q,
            K=K,
            V=V,
            rev_edge_indices=rev_edge_indices,
            size=(h_node.size(0), h_node.size(0)),
        ) # Ritorna [num_edges, hidden_dim]

        # 4. Aggiornamento delle feature di nodo aggregando i semiarchi incidenti
        node_agg = torch.zeros_like(h_node)
        node_agg.index_add_(0, col, m_edge)

        node_input = torch.cat([h_node, node_agg], dim=-1)
        h_node_new = h_node + self.node_update_mlp(node_input)
        h_edge_new = h_edge + m_edge

        return h_node_new, h_edge_new
"""
    def message(
        self,
        Q_i: torch.Tensor, # [incidents_edge, heads, head_dim] (arco ricevente)
        K_j: torch.Tensor, # [incidents_edge, heads, head_dim] (arco mittente)
        V_j: torch.Tensor,      
        rev_edge_indices_j: torch.Tensor, # indice dell'arco inverso per il mittente j  [incidents_edge]
        index: torch.Tensor,              # indice i dell'arco ricevente                [incidents_edge]
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:

        incidents_edge = Q_i.size(0)

        # 3D reshape per le teste multi-head 
        Q_i = Q_i.view(incidents_edge, self.heads, self.head_dim)
        K_j = K_j.view(incidents_edge, self.heads, self.head_dim)
        V_j = V_j.view(incidents_edge, self.heads, self.head_dim)
        
        # calcolo dell'attenzione scalare per testa [num_edges, heads]
        scores = (Q_i * K_j).sum(dim=-1) / (self.head_dim ** 0.5)

        # mascheramento anti backtracking:
        # se l'arco mittente j è l'inverso dell'arco ricevente i, annulliamo l'attenzione.
        is_backtracking = (rev_edge_indices_j == index).unsqueeze(-1)  # [incidents_edge, 1]
        scores = scores.masked_fill(is_backtracking, -1e9)
        
        alpha = softmax(scores, index, ptr, num_nodes=size_i)
        out = V_j * alpha.unsqueeze(-1) # [incidents_edge, heads, head_dim]
        # Ricomponiamo le teste
        out = out.reshape(incidents_edge, self.hidden_dim)  
        return self.out_proj(out)
"""

class DMPNNNet(nn.Module):
    """Rete Directed Message Passing Neural Network (D-MPNN) rigorosa.
    Caratteristiche scientifiche:
      - Modellazione esplicita dei semiarchi direzionati (u -> v).
      - Prevenzione del backtracking tramite mascheramento dell'arco inverso (v -> u).
      - Attenzione Multi-Head sui legami orientati.
      - Pooling combinato (Mean + Max) sia su nodi che su archi.
      - Integrazione nativa con Fingerprint di Morgan (es. 16 bit).
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int,
        num_tasks: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_morgan_fingerprint: bool = False,
        fingerprint_n_bits: int = 16,
        backend_config: dict = {}
    ) -> None:
        if backend_config.get("name", 'classical') != 'classical': 
            raise NotImplementedError("DMPNN implement only classical hardware")
        
        super().__init__()
        self.dropout = dropout
        self.use_morgan_fingerprint = use_morgan_fingerprint

        # Encoders per nodi e legami
        self.node_embed = nn.Linear(num_node_features, hidden_dim)
        self.edge_embed = nn.Linear(num_edge_features, hidden_dim)

        # Layer D-MPNN orientati
        self.layers = nn.ModuleList([
            DirectedMessagePassingLayer(hidden_dim, hidden_dim, heads=heads)
            for _ in range(num_layers)
        ])

        # Pooling combinato Mean + Max (su nodi) = hidden_dim * 2
        pool_dim = hidden_dim * 2
        self.lin_in = pool_dim + fingerprint_n_bits if use_morgan_fingerprint else pool_dim

        self.fc1 = nn.Linear(self.lin_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if x.dim() == 2 and x.size(1) == 1:
            x = x.float()

        # Inizializzazione embedding
        h_node = self.node_embed(x)
        h_edge = self.edge_embed(edge_attr)

        # Pre-calcolo del mapping dei semiarchi inversi per evitare il backtracking
        rev_edge_indices = find_reverse_edge_indices(edge_index)

        # Passaggio nei layer D-MPNN
        for layer in self.layers:
            h_node, h_edge = layer(h_node, h_edge, edge_index, rev_edge_indices)
            h_node = F.relu(h_node)
            h_node = F.dropout(h_node, p=self.dropout, training=self.training)

        # Pooling Mean + Max sugli atomi della molecola
        x_mean = global_mean_pool(h_node, batch)
        x_max = global_max_pool(h_node, batch)
        graph_repr = torch.cat([x_mean, x_max], dim=-1)

        # Fusione opzionale con la Fingerprint di Morgan
        if self.use_morgan_fingerprint and hasattr(data, "fingerprint"):
            fp = data.fingerprint
            if fp.dim() == 1:
                batch_size = batch.max().item() + 1 if batch is not None else 1
                fp = fp.view(batch_size, -1)
            x_out = torch.cat([graph_repr, fp], dim=-1)
        else:
            x_out = graph_repr

        x_out = F.relu(self.fc1(x_out))
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        logits = self.fc2(x_out)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Logits non finiti nel forward di DMPNNNet")

        return logits