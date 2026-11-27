from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from app.core.configs.configger import PipelineConfigger
from app.core.models.nets.layers.pulser_classifier_layer import PulserClassifierLayer
from app.core.models.nets.surrogate_pulser import SurrogatePulserNet

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
        surrogate_net: SurrogatePulserNet,
        dropout: float = 0.1,
        use_morgan_fingerprint: bool = False,
        fingerprint_n_bits: int = 256,
        backend_config: dict = {},
        n_heavy_max: int = 20,
        use_hybrid: bool = True,
        use_quantum_fingerprint: bool = False
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_morgan_fingerprint = use_morgan_fingerprint
        self.use_quantum_fingerprint = use_quantum_fingerprint
        self.max_nodes = n_heavy_max
        # encoding iniziale per i nodi del grafo in: num_node_features out: hidden_dim
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.surrogate_net = surrogate_net
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
        self.quantum_params_config = backend_config.get("params", {})
        
        # raddoppia dimensione (avg e max pooling) per il layer finale
        self.lin_in = hidden_dim * heads * 2
        # Se usiamo la fingerprint, sommiamo la sua dimensione al layer di input denso
        if use_morgan_fingerprint:
            self.lin_in += fingerprint_n_bits
        if self.backend_name != 'classical' and use_quantum_fingerprint:
            self.num_qubits = self.quantum_params_config["max_qubits"]
            self.lin_in += (self.num_qubits + (self.num_qubits * (self.num_qubits - 1)) // 2)
        self.fc1 = nn.Linear(self.lin_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)
        if self.backend_name != 'classical':
            # nel caso quantistico non c'è pooling, ogni nodo prepara dei parametri quantistici
            node_embedding_dim = hidden_dim * heads
        
            self.current_epoch = 0
            self.warmup_epochs = self.quantum_params_config.get("warmup_epochs", 10)
            self.waves_size = self.quantum_params_config.get("waves_size", 1)
            self.waves_param = self.quantum_params_config.get("waves_param", 1)
            self.phase_param_size = self.quantum_params_config.get("phase_param_size", 1)

            self.params_per_node = 2*self.waves_param + self.phase_param_size
            self.num_qubits =  self.quantum_params_config["max_qubits"]
            # dimensione uscita quantum 
            self.quantum_dim = self.num_qubits + (self.num_qubits * (self.num_qubits - 1)) // 2
            logger.info(f"qubits: {self.num_qubits}")
            self.max_amplitude = self.quantum_params_config.get("max_amplitude", 12.556)
            self.max_detuning = self.quantum_params_config.get("max_detuning", 125.56)
            
            # LayerNorm per stabilizzare la scala delle feature della GNN
            self.feature_norm = nn.LayerNorm(node_embedding_dim)

            # applicate dopo le onde cheby quindi waves_size
            self.amp_norm = nn.LayerNorm(self.waves_size)
            self.det_norm = nn.LayerNorm(self.waves_size)

            #TODO: implementare altri metodi di riduzione dimensionale (es VAE/VQ-VAE)
            # layer di proiezione che adatta output GNN 
            # ai parametri di controllo richiesti da Pulser
            self.node_feature_projector = nn.Linear(node_embedding_dim, self.params_per_node)

            if self.waves_size <= 1:
                # inizializziamo il bias con un valore positivo (es. 1.0) per partire con laser accesi
                nn.init.constant_(self.node_feature_projector.bias, 1.0)
            else:
                nn.init.normal_(self.node_feature_projector.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.node_feature_projector.bias)

                self.define_waves_form()

            
            self.position_layer_dim = 64

            # mlp che mappa informazioni globali del grafo molecolare c
            self.position_projector = nn.Sequential(
                nn.Linear(self.lin_in, self.position_layer_dim),
                nn.ReLU(),
                nn.Linear(self.position_layer_dim, self.num_qubits * 2) # 2 coordinate (x, y) per ciascun qubit
            )

            self.qvc_classifier = PulserClassifierLayer(
                backend_name=self.backend_name,
                max_num_nodes=self.max_nodes,
                num_qubits= self.num_qubits,
                backend_params=self.quantum_params_config,
                use_hybrid=use_hybrid,
                surrogate_net=surrogate_net
            )

            #multi label head 
            self.multi_label_head = nn.Linear(self.quantum_dim, num_tasks)
            self.fusion_head = nn.Linear(num_tasks * 2, num_tasks)


    def define_waves_form(self):
        """
        Inizializza la base di Chebyshev per la costruzione delle forme d'onda temporali.
        Da chiamare nell'__init__.
        """
        # tempo normalizzato da [0, 1] a [-1, 1] per Chebyshev
        t = torch.linspace(0, 1, self.waves_size)
        x = 2.0 * t - 1.0
        
        # matrice di base di Chebyshev [waves_size, waves_param]
        cheby_basis = torch.zeros(self.waves_size, self.waves_param)
        cheby_basis[:, 0] = 1.0  # T_0(x) = 1
        if self.waves_param > 1:
            cheby_basis[:, 1] = x  # T_1(x) = x
        if self.waves_param > 2:
            for k in range(2, self.waves_param):
                # relazione di ricorrenza: T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
                cheby_basis[:, k] = 2.0 * x * cheby_basis[:, k-1] - cheby_basis[:, k-2]

        # buffer per farlo spostare eventualmente su GPU insieme al modello
        self.register_buffer('cheby_basis', cheby_basis)

    def compute_waves_from_param(self, x_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Crea le onde a partire dalle feature dei nodi o da parametri passati in input,
        sfruttando la stessa identica logica utilizzata nel forward del modello.
        """
        # se i parametri sono scalari
        if self.waves_size <= 1:
            # normalizzare non ha senso se le features hanno domini diversi
            x_dense = self.feature_norm(x_dense)

            # proietta feature dei nodi nei parametri richiesti da Pulser
            # x_projected avrà forma: [Batch_Size, num_node_features, params_per_node]
            x_projected = self.node_feature_projector(x_dense)
            # azzeriamo via moltiplicazione le feature dove mask è False 
            # per non applicare laser a qubit associati a nodi/atomi di padding
            x_projected = x_projected * mask.unsqueeze(-1).float()
            # passa per layer non lineare positivo (frequenze di Rabi positive per definizione)
            # Aggiungiamo un LayerNorm per stabilizzare la scala delle feature della GNN
        
            #quando era solo amp
            x_projected = F.relu(abs(x_projected))
            min_base = 0.05
            x_projected = torch.clamp(x_projected, min=min_base, max=self.max_amplitude)
        elif self.waves_size % 4 == 0:
            x_projected = self.node_feature_projector(x_dense)
            x_projected = x_projected * mask.unsqueeze(-1).float()
            
            # chunking del tensore PyTorch
            amp_coeffs = x_projected[..., : self.waves_param]                     
            det_coeffs = x_projected[..., self.waves_param : 2 * self.waves_param]     
            phase_raw = x_projected[..., 2 * self.waves_param : 2 * self.waves_param + self.phase_param_size]

            # costruzione bin temporali tramite Chebyshev
            # matmul tra [..., 4] e [16, 4].T -> [..., 16]
            amp_raw_bins = torch.matmul(amp_coeffs, self.cheby_basis.T)
            det_raw_bins = torch.matmul(det_coeffs, self.cheby_basis.T)
  
            # normalizzare non ha senso se le features hanno domini diversi
            amp_raw_bins = self.amp_norm(amp_raw_bins)
            det_raw_bins = self.det_norm(det_raw_bins)
            # attivazioni fisiche corrette mantenendo il grafo di PyTorch
            amp_act = torch.sigmoid(amp_raw_bins) * self.max_amplitude  # positiva [0, max_amp]
            
            #amp_act = F.relu(abs(amp_raw_bins))
            det_act = torch.tanh(det_raw_bins) * self.max_detuning   # simmetrica [-max_det, +max_det]
            phase_act = torch.tanh(phase_raw) * 6.28 # Angolare [-2pi, +2pi] 

            # unisci i tensori attivati
            x_projected = torch.cat([amp_act, det_act, phase_act], dim=-1)
        else: 
            raise ValueError(f"Numero di parametri deve essere minore di 3 o divisibile per 4")
            
        return x_projected

    def compute_position_embeddings(self, graph_out):
                
        # posizioni dei qubits per il batch, ha già RELU dentro
        coords_flat = self.position_projector(graph_out) # [Batch, num_qubits * 2]
        qubit_positions = coords_flat.view(-1, self.num_qubits, 2) # [Batch, num_qubits, 2]
        return qubit_positions
    
    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)


        # traduce formato sparso pyg: [totale_atomi_nel_batch, hidden_dim * heads]
        # in formato denso: [batch_size, max_num_nodes, hidden_dim * heads]
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes) 
        
        # Pooling globale: media e max concatenati lungo l'asse delle feature 
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_repr = torch.cat([x_mean, x_max], dim=-1)
        extra_features = [graph_repr]
        # Fusione opzionale con la Fingerprint di Morgan
        if self.use_morgan_fingerprint and hasattr(data, "fingerprint"):
            fp = data.fingerprint
            
            # Se PyG ha appiattito la fingerprint in un tensore 1D durante il batching:
            if fp.dim() == 1:
                # Ricaviamo il batch_size effettivo contando quanti grafi unici ci sono nel batch
                batch_size = batch.max().item() + 1
                fp = fp.view(batch_size, -1)
                
            extra_features.append(fp)

        if self.use_quantum_fingerprint and hasattr(data, "quantum_fingerprint"):
            q_fp = data.quantum_fingerprint
            if q_fp.dim() == 1:
                batch_size = batch.max().item() + 1
                q_fp = q_fp.view(batch_size, -1)
            extra_features.append(q_fp)

        graph_out = torch.cat(extra_features, dim=-1)
        x = F.relu(self.fc1(graph_out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        classical_logits = self.fc2(x)
        if self.backend_name == 'classical':
            #ritorna 
            if not torch.isfinite(classical_logits).all():
                raise RuntimeError("Logits non finiti nel forward di GATv2Net")
            return classical_logits
        else:
            
            qubit_positions = self.compute_position_embeddings(graph_out)
            x_projected = self.compute_waves_from_param(
                x_dense=x_dense, 
                mask=mask
            )
            
            qubit_probs = self.apply_quantum_layer(x_projected, mask, qubit_positions)

            quantum_logits = self.multi_label_head(qubit_probs)
            combined = torch.cat([quantum_logits, classical_logits], dim=-1)
            logits = self.fusion_head(combined)  # [batch_size, C]
            return logits

    def apply_quantum_layer(self, x_projected, mask, qubit_positions):
         # tensore al layer Pulser (simulazione per ogni molecola nel batch)
        # il layer Pulser restituisce direttamente l'output [Batch_Size, 1] pronto per la loss binaria
        quantum_out = self.qvc_classifier(x_projected, mask=mask, qubit_positions=qubit_positions)
        logger.debug(f"quantum classifier output: {quantum_out}")
        
        return quantum_out