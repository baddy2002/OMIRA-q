
from pulser.register import RectangularLatticeLayout
import torch.nn as nn
import logging 
import sys 

from pulser.devices import AnalogDevice

from app.core.models.nets.functions.pulser_function import PulserFunction

stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler],
)
logger = logging.getLogger("PulserClassifierLayer")


class PulserClassifierLayer(nn.Module):
    def __init__(self, backend_name: str, num_atoms: int, backend_params: dict):
        super().__init__()
        self.backend_name = backend_name
        self.num_atoms = num_atoms
        self.backend_params = backend_params

        # flag per il rumore dai parametri passati via YAML
        self.use_noise = backend_params.get("use_noise", False)
        self.min_amplitude = backend_params.get("min_amplitude", 0.0)
        self.max_amplitude = backend_params.get("max_amplitude", 20.0)
        self.max_atoms = backend_params.get("max_atoms", 20)  # limite massimo di atomi per sicurezza

        #TODO: implementare mapping conforme alla forma di molecole
        #TODO: conviene mappare solo 6 qubit e prendere output come classificatore binario per le 6 label
        # AnalogDevice vuole avere almeno la metà degli slot vuoti (filling fraction = 0.5)
        total_traps = self.num_atoms * 2
        
        # calcoliamo righe e colonne per disporre tutti i max_atoms 
        cols = self.max_atoms // 4 if self.max_atoms >= 4 else 4
        rows = (total_traps + cols - 1) // cols
        
        # layout reticolare compatibile con AnalogDevice
        # ogni atomo è distanziato di una distanza fisica sicura (6 micrometri)
        layout = RectangularLatticeLayout(rows=rows, columns=cols, col_spacing=6.0, row_spacing=6.0)

        # Selezioniamo i primi `num_atoms` siti dal layout per formare il registro
        trap_ids = list(range(num_atoms))
        qubit_ids = [f"atom_{i}" for i in range(num_atoms)]
        
        self.register = layout.define_register(*trap_ids, qubit_ids=qubit_ids)

        self.device = AnalogDevice
        logger.debug(f"Device specifications: {self.device.specs}")

    def forward(self, x_projected, mask=None):
        """
        x_projected: tensore in output dal proiettore lineare della GNN
        Forma attesa: [Batch_Size, num_atoms, params_per_atom]
        """
        if mask is None:
            raise ValueError("Necessario indicare una maschera validità non nulla")

        logger.info(f"valore di x_projected nel forward di PulserClassifierLayer: {x_projected} ")
                
        return PulserFunction.apply(
            x_projected, mask, self.register, self.device, 
            self.backend_name, self.use_noise, self.backend_params
        ) 