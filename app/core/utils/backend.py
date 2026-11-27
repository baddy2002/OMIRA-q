# classical

import torch

def get_optimal_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilizzo GPU NVIDIA (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Utilizzo GPU Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Utilizzo CPU")
    return device

#Pulser 

#Bloqade