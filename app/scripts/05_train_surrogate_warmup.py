import logging
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


from app.core.configs.configger import PipelineConfigger
from app.core.models.nets.gatv2 import GATv2Net
from app.core.models.nets.functions.pulser_function import PulserFunction
from app.core.models.nets.layers.pulser_classifier_layer import PulserClassifierLayer
from app.core.models.nets.surrogate_pulser import SurrogatePulserNet

logger = logging.getLogger(__name__)
configger = PipelineConfigger()

def generate_random_node_features(batch_size: int, num_nodes: int, num_node_features: int, device: torch.device):
    """
    Genera feature casuali per i nodi del grafo che rispecchiano la scala 
    di input che la GATv2 riceve solitamente in pasto.
    """
    return torch.randn(batch_size, num_nodes, num_node_features, device=device)

def generate_dataset(model_name: str = "gatv2", backend_name: str = "neutral_atom_sim", num_samples: int = 2500):
    logger.info("Avvio generazione dataset per il modello surrogato Pulser...")
    
    gcfg = configger.get_model_config(model_name)
    bcfg = configger.get_backend_config(backend_name)
    p = gcfg["params"]
    
    device_torch = torch.device("cpu") # Eseguiamo su CPU per la generazione offline pulita
    
    # Istanziamo la GATv2 per sfruttare il suo proiettore di feature e le funzioni di onda
    gatv2 = GATv2Net(
        num_node_features=p["num_node_features"],
        num_edge_features=p["num_edge_features"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        heads=p["heads"],
        num_tasks=p["num_tasks"],
        dropout=p["dropout"],
        use_morgan_fingerprint=p.get("use_morgan_fingerprint", False),
        fingerprint_n_bits=p.get("fingerprint_n_bits", 256),
        backend_config=bcfg,
        use_hybrid=False,
        surrogate_net=None,
        use_quantum_fingerprint=False
    ).to(device_torch)
    gatv2.eval()

    # Recuperiamo il layer Pulser sottostante per accedere a register, device e parametri backend
    pulser_layer: PulserClassifierLayer = gatv2.qvc_classifier
    device_pulser = pulser_layer.device
    backend_params = pulser_layer.backend_params
    waves_size = pulser_layer.waves_size
    phase_param_size = pulser_layer.phase_param_size
    num_qubits = pulser_layer.num_qubits
    max_origin_dist = backend_params.get("qubit_max_origin_distance", 38.0)
    min_dist = backend_params.get("qubit_min_distance", 5.0)

    dataset_waves = []
    dataset_pos = []
    dataset_Y = []

    logger.info(f"Generazione di {num_samples} campioni con geometrie di qubit dinamiche...")

    with torch.no_grad():
        for i in range(num_samples):
            batch_size = 1
            max_nodes = pulser_layer.max_num_nodes
            num_nodes_in_sample = np.random.randint(8, max_nodes + 1)
            
            x_raw = generate_random_node_features(
                batch_size=1, 
                num_nodes=num_nodes_in_sample, 
                num_node_features=p["num_node_features"], 
                device=device_torch
            ).squeeze(0)
            batch = torch.zeros(num_nodes_in_sample, dtype=torch.long, device=device_torch) # Tutti nel grafo 0
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device_torch) # Eventuali archi vuoti o generati
            fingerpints = torch.eye()
            # 2. Eseguiamo la passata attraverso encoder e convoluzioni della GAT (esattamente come nel forward)
            x = gatv2.node_encoder(x_raw)
            for conv in gatv2.convs:
                x = conv(x, edge_index, edge_attr=None)
                x = F.relu(x)
                x = F.dropout(x, p=gatv2.dropout, training=gatv2.training)

            # 3. Otteniamo le posizioni dei qubit con la tua funzione esatta
            qubit_coords = gatv2.compute_position_embeddings(x, batch, fingerpints) # [1, num_qubits, 2]

            # 4. Creiamo x_dense e mask usando to_dense_batch
            x_dense, mask = to_dense_batch(x, batch, max_num_nodes=max_nodes)

            # 5. Calcoliamo le onde proiettate
            x_projected = gatv2.compute_waves_from_param(x_dense=x_dense, mask=mask)
            # 3. Costruiamo il register dinamico per Pulser
            try:

                qubit_probs_list = PulserFunction.simulate_single(
                    i=0,
                    x_projected=x_projected,
                    mask=mask,
                    device=device_pulser,
                    backend_params=backend_params,
                    waves_size=waves_size,
                    phase_param_size=phase_param_size,
                    backend_name=backend_name,
                    noise_model=None,
                    num_qubits=num_qubits,
                    qubit_max_origin_distance=max_origin_dist,
                    qubit_min_distance=min_dist,
                    qubit_positions_embedding=qubit_coords,
                    max_layout_fill_factor=0.5
                )
                
                dataset_waves.append(x_projected.detach().cpu())
                dataset_pos.append(qubit_coords[0].clone().detach().cpu())
                dataset_Y.append(torch.tensor(qubit_probs_list, dtype=torch.float32))

            except Exception as e:
                logger.warning(f"Salto il campione {i} a causa di un errore nella simulazione: {e}")
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"Completati {i + 1}/{num_samples} campioni.")

    if len(dataset_waves) > 0:
        waves_tensor = torch.stack(dataset_waves)
        pos_tensor = torch.stack(dataset_pos)
        Y_tensor = torch.stack(dataset_Y)

        output_path = Path("data/input/surrogate_dataset.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({"waves": waves_tensor, "positions": pos_tensor, "Y": Y_tensor}, output_path)
        logger.info(f"Dataset geometrico salvato in {output_path} | Onde: {waves_tensor.shape}, Pos: {pos_tensor.shape}, Target: {Y_tensor.shape}")
    else:
        logger.error("Nessun campione generato con successo!")

def train_surrogate_model(num_samples: int = 2500, data_path: Path = Path("data/input/surrogate_dataset.pt"), epochs: int = 50):
    logger.info("Avvio addestramento del modello surrogato...")
    
    # 1. Carichiamo il dataset salvato
    data = torch.load(data_path)
    waves = data["waves"] 
    positions = data["positions"] 
    Y = data["Y"]

    if waves.ndim == 4:
        waves = waves.squeeze(1)# Diventa [200, 20, 33]
        
    # Appiattiamo le feature dei nodi per darli in pasto all'MLP: [Batch, Nodes * Params]
    batch_size, num_nodes, params_per_node = waves.shape
    waves_flat = waves.reshape(batch_size, num_nodes * params_per_node)
    pos_flat = positions.reshape(batch_size, -1)
    
    # Dimensione di input totale = features delle onde + coordinate x,y dei qubit
    input_dim = waves_flat.shape[1] + pos_flat.shape[1]
    num_qubits = Y.shape[1]

    # 2. Creazione DataLoader
    dataset = TensorDataset(waves_flat, pos_flat, Y)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    surrogate_net = SurrogatePulserNet(input_dim=input_dim, num_qubits=num_qubits, hidden_dim=128)
    optimizer = torch.optim.AdamW(surrogate_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    logger.info("Addestramento del modello surrogato con Train/Test split...")
    for epoch in range(epochs):
        surrogate_net.train()
        train_loss = 0.0
        for b_waves, b_pos, by in train_loader:
            optimizer.zero_grad()
            preds = surrogate_net(b_waves, b_pos)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        surrogate_net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for b_waves, b_pos, by in test_loader:
                preds = surrogate_net(b_waves, b_pos)
                test_loss += criterion(preds, by).item()
                
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.5f} | Test Loss: {test_loss/len(test_loader):.5f}")
            
    weights_path = Path("data/output/models/surrogate_weights.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surrogate_net.state_dict(), weights_path)
    logger.info(f"Modello surrogato geometrico addestrato e salvato in {weights_path}")

def main():
    generate_dataset("gatv2", "neutral_atom_sim", num_samples=6000) 
    train_surrogate_model(num_samples=6000)
if __name__ == "__main__":
    main()