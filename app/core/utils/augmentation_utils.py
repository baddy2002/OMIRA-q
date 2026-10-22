# app/core/utils/augmentation_utils.py
import torch
from torch_geometric.data import Dataset, Data

class AugmentedPyGDataset(Dataset):
    """Dataset wrapper che mantiene i dati originali E genera N copie ruotate al volo ad ogni epoca."""
    def __init__(self, data_list: list[Data], num_aug_per_mol: int = 1, jitter_std: float = 0.005):
        super().__init__()
        self.original_data = data_list
        self.num_aug = num_aug_per_mol  # es. 1 -> Raddoppia i dati (1 originale + 1 ruotata)
        self.jitter_std = jitter_std

    def len(self) -> int:
        # Se num_aug = 1, la lunghezza raddoppia: N_orig * (1 + 1)
        return len(self.original_data) * (1 + self.num_aug)

    def get(self, idx: int) -> Data:
        orig_idx = idx // (1 + self.num_aug)
        aug_type = idx % (1 + self.num_aug)

        # Cloniamo l'oggetto Data originale
        data = self.original_data[orig_idx].clone()

        # aug_type == 0 -> Molecola ORIGINALE CANONICA (senza alcuna modifica)
        if aug_type > 0 and hasattr(data, "pos") and data.pos is None:
            # Molecola TRASFORMATA: Rotazione SO(3) e micro-jittering indipendente
            A = torch.randn(3, 3)
            Q, _ = torch.linalg.qr(A)
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            
            # Rotazione indipendente
            data.pos = torch.matmul(data.pos, Q)
            
            # Micro-jittering
            if self.jitter_std > 0:
                data.pos = data.pos + torch.randn_like(data.pos) * self.jitter_std

        return data