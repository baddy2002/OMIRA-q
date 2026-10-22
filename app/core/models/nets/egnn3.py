from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import scatter, softmax


class EGATLayer(nn.Module):
    """Layer di E(n)-Equivariant Graph Attention Network (EGAT).
    Combina l'invarianza/equivarianza geometrica 3D delle EGNN con il meccanismo di attenzione.
    """

    def __init__(self, hidden_dim: int, num_edge_features: int) -> None:
        super().__init__()
        
        # MLP per il calcolo del messaggio combinando feature di nodo, distanza radiale e attributi arco
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + num_edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # MLP per il calcolo dei coefficienti di attenzione (stile GAT applicato al messaggio geometrico)
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        # MLP per il peso di aggiornamento delle coordinate (mantiene l'equivarianza E(n))
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
        # MLP per l'aggiornamento delle feature di nodo
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index
        
        # distanza euclidiana quadrata (Invariante per rotazioni e traslazioni)
        coord_diff = pos[row] - pos[col]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)
        radial = torch.clamp(radial, min=1e-8, max=1e4)

        # concatena feature sorgente, destinazione, distanza radiale e attributi del legame (arco)
        edge_feat = torch.cat([h[row], h[col], radial, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_feat)

        # calcolo dei pesi di attenzione (Attention Coefficients)
        raw_attn = self.attn_mlp(m_ij)
        alpha = softmax(raw_attn, row, num_nodes=h.size(0))  # Normalizzazione softmax sui vicini del nodo target (row)

        # aggiornamento coordinate (Equivariante E(n))
        # Il movimento spaziale viene pesato sia dall'MLP geometrico sia dai coefficienti di attenzione
        coord_weight = self.coord_mlp(m_ij)
        coord_weight = torch.clamp(coord_weight, min=-10.0, max=10.0)
        pos_update = coord_diff * coord_weight * alpha
        pos_agg = scatter(pos_update, row, dim=0, dim_size=h.size(0), reduce="sum")
        pos = pos + pos_agg

        # aggiornamento feature di nodo con messaggi pesati dall'attenzione
        m_ij_attn = m_ij * alpha
        agg = scatter(m_ij_attn, row, dim=0, dim_size=h.size(0), reduce="sum")
        
        node_feat = torch.cat([h, agg], dim=-1)
        h = h + self.node_mlp(node_feat)  # Connessione residua

        return h, pos
"""
def apply_on_the_fly_3d_augmentation(batch, jitter_std: float = 0.01) -> None:
    Applica una rotazione casuale SO(3) e un micro-jittering gaussiano al volo sul batch di PyG.
    if not hasattr(batch, "pos") or batch.pos is None:
        return

    # Genera matrice di rotazione SO(3) casuale per l'intero batch (preserva la geometria interna)
    A = torch.randn(3, 3, device=batch.pos.device)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Ruota le coordinate di tutti gli atomi
    batch.pos = torch.matmul(batch.pos, Q)

    # Aggiungi un piccolissimo jittering gaussiano (vibrazione termica artificiale)
    if jitter_std > 0:
        jitter = torch.randn_like(batch.pos) * jitter_std
        batch.pos = batch.pos + jitter
"""

class EGNNNet(nn.Module):
    """E(n)-Equivariant Graph Attention Network (EGAT) completa per il task NCI-60.
    Supporta:
      - Feature di nodo e di arco (bond features).
      - Meccanismo di attenzione geometrica nei layer.
      - Pooling combinato (Mean + Max).
      - Fusione opzionale con la Fingerprint di Morgan (configurabile da YAML).
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int,
        num_tasks: int,
        dropout: float = 0.1,
        use_morgan_fingerprint: bool = False,
        fingerprint_n_bits: int = 16,
        backend_config: dict = {}
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_morgan_fingerprint = use_morgan_fingerprint

        # Encoder dei nodi
        self.node_embed = nn.Linear(num_node_features, hidden_dim)

        # Stack di layer EGAT equivarianti
        self.layers = nn.ModuleList([
            EGATLayer(hidden_dim, num_edge_features) for _ in range(num_layers)
        ])

        # Dimensione base derivata dal pooling combinato (Mean + Max raddoppia la hidden_dim)
        pool_dim = hidden_dim * 2
        
        # Fusione opzionale con la Fingerprint di Morgan
        self.lin_in = pool_dim + fingerprint_n_bits if use_morgan_fingerprint else pool_dim
        if backend_config.get("name", "classical") == 'classical':
            self.fc1 = nn.Linear(self.lin_in, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data) -> torch.Tensor:

        h, pos, edge_index, edge_attr, batch = (
            data.x,
            data.pos,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        
        if h.dim() == 2 and h.size(1) == 1:
            h = h.float()

        h = self.node_embed(h)

        # Passaggio attraverso i layer EGAT (aggiorna sia le feature h che le coordinate pos preservando l'equivarianza)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Pooling combinato Mean + Max (esattamente come in GATv2)
        x_mean = global_mean_pool(h, batch)
        x_max = global_max_pool(h, batch)
        graph_repr = torch.cat([x_mean, x_max], dim=-1)

        # Fusione opzionale con la Fingerprint di Morgan
        if self.use_morgan_fingerprint and hasattr(data, "fingerprint"):
            fp = data.fingerprint
            
            # Se PyG ha appiattito la fingerprint in un tensore 1D durante il batching:
            if fp.dim() == 1:
                # Ricaviamo il batch_size effettivo contando quanti grafi unici ci sono nel batch
                batch_size = batch.max().item() + 1
                fp = fp.view(batch_size, -1)
                
            x = torch.cat([graph_repr, fp], dim=-1)
        else:
            x = graph_repr

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc2(x)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Logits non finiti nel forward di EGNNNet (EGAT)")
            
        return logits