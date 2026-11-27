# app/core/utils/augmentation_utils.py
import torch
from torch_geometric.data import Dataset, Data

class AugmentedPyGDataset(Dataset):
    """
    dataset wrapper che mantiene i dati originali e 
    genera  N copie ruotate al volo ad ogni epoca.
    """
    def __init__(self, data_list: list[Data], num_aug_per_mol: int = 1, jitter_std: float = 0.005):
        super().__init__()
        self.original_data = data_list
        self.num_aug = num_aug_per_mol  # es. 1 -> Raddoppia i dati (1 originale + 1 ruotata)
        self.jitter_std = jitter_std

    def len(self) -> int:
        return len(self.original_data) * (1 + self.num_aug)

    def get(self, idx: int) -> Data:
        """
        PyTorch Geometric chiama automaticamente il metodo __getitem__ della classe base quando 
        accediamo ad un elemento del dataset o quando un dataloader preleva un batch
        il metodo __getitem__ chiama automaticamente questo metodo
        """
        orig_idx = idx // (1 + self.num_aug)
        aug_type = idx % (1 + self.num_aug)

        # cloniamo l'oggetto Data originale
        data = self.original_data[orig_idx].clone()

        # aug_type == 0 -> Molecola ORIGINALE CANONICA (senza alcuna modifica)
        if aug_type > 0 and hasattr(data, "pos") and data.pos is not None:
            # molecola trasformata: Rotazione SO(3) e micro-jittering indipendente
            A = torch.randn(3, 3)
            # QR decomposition per ottenere una matrice di rotazione casuale
            Q, _ = torch.linalg.qr(A)
            # importante: dal punto di vista chimico una riflessione/inversione O(3) 
            # cambia la chiralità della molecola, quindi Se det(Q) < 0, 
            # invertiamo il segno della prima colonna per ottenere una rotazione pura.
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            
            # rotazione indipendente
            data.pos = torch.matmul(data.pos, Q)
            
            # micro-jittering
            if self.jitter_std > 0:
                data.pos = data.pos + torch.randn_like(data.pos) * self.jitter_std

        return data