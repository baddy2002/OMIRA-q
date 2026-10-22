from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from app.core.configs.configger import PipelineConfigger
from app.core.models.nets.layers.pulser_classifier_layer import PulserClassifierLayer

configger = PipelineConfigger()

stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler],
)
logger = logging.getLogger("GatV2Net")

class GATv2Net(nn.Module):
    """Rete GATv2 con feature di nodo e di arco continue."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        num_tasks: int,
        dropout: float = 0.1,
        use_morgan_fingerprint: bool = False,
        fingerprint_n_bits: int = 256,
        backend_config: dict = {}
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_morgan_fingerprint = use_morgan_fingerprint

        # encoding iniziale per i nodi del grafo in: num_node_features out: hidden_dim
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                hidden_dim, hidden_dim, heads=heads, concat=True,
                add_self_loops=True, edge_dim=num_edge_features,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                    add_self_loops=True, edge_dim=num_edge_features,
                )
            )

        # testa differente in base al backend target (quantum non ha pooling)
        self.backend_name = backend_config.get("name", "classical")

        if self.backend_name == 'classical':
            # raddoppia dimensione (avg e max pooling) per il layer finale
            pool_dim = hidden_dim * heads * 2
            # Se usiamo la fingerprint, sommiamo la sua dimensione al layer di input denso
            self.lin_in = pool_dim + fingerprint_n_bits if use_morgan_fingerprint else pool_dim

            self.fc1 = nn.Linear(self.lin_in, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_tasks)
        else:
            # nel caso quantistico non c'è pooling, ogni nodo prepara un qubit
            node_embedding_dim = hidden_dim * heads

            self.quantum_params_config = backend_config.get("params", {})
            params_per_atom = self.quantum_params_config.get("params_per_atom", 1)
            self.num_atoms = self.quantum_params_config.get("max_atoms", 20)
            self.max_amplitude = self.quantum_params_config.get("max_amplitude", 12.556)
            # LayerNorm per stabilizzare la scala delle feature della GNN
            self.feature_norm = nn.LayerNorm(node_embedding_dim)

            #TODO: implementare altri metodi di riduzione dimensionale (es VAE/VQ-VAE)
            # layer di proiezione che adatta output GNN 
            # ai parametri di controllo richiesti da Pulser
            self.node_feature_projector = nn.Linear(node_embedding_dim, self.num_atoms)

            # Inizializziamo il bias con un valore positivo (es. 1.0) per partire con laser accesi
            nn.init.constant_(self.node_feature_projector.bias, 1.0)

            self.qvc_classifier = PulserClassifierLayer(
                backend_name=self.backend_name,
                num_atoms= self.num_atoms,
                backend_params=self.quantum_params_config
            )

            # TODO: gestire multilabel direttamente dentro layer quantistico ??
            #multi label head 
            self.multi_label_head = nn.Linear(self.num_atoms, 6)


    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.backend_name == 'classical':
            # Pooling globale: media e max concatenati lungo l'asse delle feature 
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
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
                raise RuntimeError("Logits non finiti nel forward di GATv2Net")
            return logits
        else:
            from torch_geometric.utils import to_dense_batch
            
            # tensore dei nodi in formato denso [batch_size, max_num_nodes, embedding_dim]
            # num_node_features come padding/troncamento standard
            x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.num_atoms) 
            # x_dense : [Batch_Size, num_node_features, hidden_dim * heads]
            
            x_dense = self.feature_norm(x_dense)

            # proietta feature dei nodi nei parametri richiesti da Pulser
            # x_projected avrà forma: [Batch_Size, num_node_features, params_per_atom]
            x_projected = self.node_feature_projector(x_dense)
            # azzeriamo via moltiplicazione le feature dove mask è False 
            # per non applicare laser a qubit associati a nodi/atomi di padding
            x_projected = x_projected * mask.unsqueeze(-1).float()
            # passa per layer non lineare positivo (frequenze di Rabi positive per definizione)
            # Aggiungiamo un LayerNorm per stabilizzare la scala delle feature della GNN
            
            #scale_factor = 5.0
            min_base = 0.05
            #x_projected = F.leaky_relu(x_projected, negative_slope=0.01) * scale_factor
            
            x_projected = F.relu(abs(x_projected))
            x_projected = torch.clamp(x_projected, min=min_base, max=self.max_amplitude)
            # tensore al layer Pulser (simulazione per ogni molecola nel batch)
            # il layer Pulser restituisce direttamente l'output [Batch_Size, 1] pronto per la loss binaria
            quantum_out = self.qvc_classifier(x_projected, mask=mask)
            logger.debug(f"quantum classifier output: {quantum_out}")
            logits = self.multi_label_head(quantum_out)
            return logits