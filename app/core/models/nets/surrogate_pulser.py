import torch
import torch.nn as nn

class SurrogatePulserNet(nn.Module):
    """
    Modello surrogato (MLP) che approssima la simulazione quantistica di Pulser.
    Prende in input sia i parametri delle onde (Ampiezza, Detuning, Fase) 
    sia le coordinate spaziali dei qubit (x, y).
    e restituisce le probabilità dei qubit.
    """
    def __init__(self, input_dim: int = 0, num_qubits: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.num_qubits = num_qubits
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_qubits),
            nn.Hardsigmoid()  # Restituisce probabilità tra 0 e 1 per ciascun qubit
        )

    def forward(self, waves: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: tensore dei parametri delle onde generati dalla GNN (es. Chebyshev / CustomWaveform)
        # Assicuriamoci che i tensori siano appiattiti correttamente per la concatenazione
        if waves.ndim > 2:
            waves = waves.reshape(waves.shape[0], -1)
        if positions.ndim > 2:
            positions = positions.reshape(positions.shape[0], -1)

        # concatena onde e posizioni
        x = torch.cat([waves, positions], dim=-1)
        return self.net(x)