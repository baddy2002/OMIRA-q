
from pulser.register import RectangularLatticeLayout
import torch.nn as nn
import logging 
import sys 

from pulser.devices import AnalogDevice

from app.core.models.nets.functions.hybrid_pulser_function import HybridPulserFunction
from app.core.models.nets.functions.pulser_function import PulserFunction
from app.core.models.nets.surrogate_pulser import SurrogatePulserNet

stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler],
)
logger = logging.getLogger("PulserClassifierLayer")


class PulserClassifierLayer(nn.Module):
    def __init__(self, backend_name: str, max_num_nodes: int, num_qubits: int, backend_params: dict, use_hybrid: bool = True, surrogate_net: SurrogatePulserNet = None):
        super().__init__()
        self.backend_name = backend_name
        self.max_num_nodes = max_num_nodes
        self.num_qubits = num_qubits
        self.backend_params = backend_params
        self.surrogate_net = surrogate_net
        self.use_hybrid = use_hybrid

        # flag per il rumore dai parametri passati via YAML
        self.use_noise = backend_params.get("use_noise", False)
        self.min_amplitude = backend_params.get("min_amplitude", 0.0)
        self.max_amplitude = backend_params.get("max_amplitude", 20.0)
        self.max_atoms = backend_params.get("max_atoms", 20)  # limite massimo di atomi per sicurezza
        self.waves_size = backend_params.get("waves_size", 1)
        self.waves_param = backend_params.get("waves_param", 1)
        self.phase_param_size = backend_params.get("phase_param_size", 1)

        self.params_per_node = 2*self.waves_param + self.phase_param_size

        self.device = AnalogDevice
        logger.debug(f"Device specifications: {self.device.specs}")

    def forward(self, x_projected, mask=None, qubit_positions=None):
        """
        x_projected: tensore in output dal proiettore lineare della GNN
        Forma attesa: [Batch_Size, num_atoms, params_per_node]
        """
        if mask is None:
            raise ValueError("Necessario indicare una maschera validità non nulla")

        if self.use_hybrid: 
            return HybridPulserFunction.forward(
                x_projected, mask, self.surrogate_net, self.device, 
                self.backend_name, self.use_noise, self.backend_params, 
                self.waves_size, self.phase_param_size, qubit_positions
            ) 
        else :
            return PulserFunction.forward(
                x_projected, mask, self.device, 
                self.backend_name, self.use_noise, self.backend_params, 
                self.waves_size, self.phase_param_size, qubit_positions
            )