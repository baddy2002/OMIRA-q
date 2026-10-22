import torch
import torch.nn as nn
import numpy as np
import logging
import concurrent.futures

from pulser import Sequence, Pulse
from pulser.backend import NoiseModel
from pulser.backend import BitStrings
from pulser_simulation import QutipBackendV2, QutipConfig

from pulser.waveforms import CustomWaveform

from app.core.utils.parallel_utils import ExecutorSingleton

logger = logging.getLogger("PulserClassifierLayer")

class PulserFunction(torch.autograd.Function):
    log_printed = False
    
    @staticmethod
    def forward(ctx, x_projected, mask, register, device, backend_name, use_noise, backend_params):
        ctx.save_for_backward(x_projected)
        ctx.mask = mask
        executor = ExecutorSingleton.get_executor(max_workers=8)
        if torch.isnan(x_projected).any() or torch.isinf(x_projected).any():
            logger.warning("ATTENZIONE: Trovati NaN o Inf nelle feature proiettate dalla GNN!")
        batch_size = x_projected.shape[0]
        min_amplitude = backend_params.get("min_amplitude", 0.0)
        max_amplitude = backend_params.get("max_amplitude", 12.556)
        num_atoms = len(register.qubits)
        logger.debug(f"numero atomi: {num_atoms}")

       # configurazione del rumore 
        noise_model = None
        if backend_name == 'neutral_atom_sim' and use_noise:
            noise_model = NoiseModel(
                state_prep_error=0.01,
                dephasing_rate=0.05,
                temperature=10.0,
                trap_waist=1.0,
                trap_depth=100.0
            )

        def _build_sequence(atom_features_sample, atom_mask_sample):
            """
            Costruisce la sequenza di impulsi Pulser mappando le feature 
            distinte di ciascun atomo della GNN sui rispettivi target del circuito.
            @param: 
                atom_features_sample: features di tutto il grafo: [num_atoms, params_per_atom]
                atom_mask_sample: mask per specifica molecola: [num_atoms]
            """ 
            seq = Sequence(register, device)
    
            # verifica la disponibilità di un canale locale o usa il canale locale di Rydberg global
            # chadoq2 dovrebbe averli
            # per sicurezza sui canali disponibili del device:
            available_ch = list(device.channels.keys())
            # cerca un canale locale, altrimenti prendi primo disponibile
            local_channel_name = None
            for ch_name, ch_obj in device.channels.items():
                if ch_obj.basis == "rydberg": # o controllo sulle capacità locali
                    local_channel_name = ch_name
                    channel_name = local_channel_name
                    if not PulserFunction.log_printed:
                        PulserFunction.log_printed = True
                        logger.info(f"canale locale trovato: {local_channel_name}")
                    break
            
            if local_channel_name is None:
                channel_name = available_ch[0]
                if not PulserFunction.log_printed:
                    PulserFunction.log_printed = True
                    logger.warning(f"canale locale non trovato, ripiegato sul primo {available_ch[0]}")
                
            # se ci sono canali locali 
            # dichiariamo il canale con un target iniziale sul primo atomo
            if local_channel_name is not None: 
                first_atom = f"atom_0"
                seq.declare_channel('local_rydberg', channel_name, initial_target=first_atom)
            else:
                # altrimenti canale globale
                seq.declare_channel('rydberg_global', channel_name)
    
            # estraiamo le feature degli atomi convertendole in numpy
            features_np = atom_features_sample.squeeze(-1).detach().cpu().numpy() # Forma: [num_atoms]
            # maschera decisionale per applicare l'impulso laser su determinato qubit
            mask_np = atom_mask_sample.detach().cpu().numpy() # [num_atoms] di booleani 

            # iteriamo gli atomi del registro e applichiamo un impulso 
            # la cui ampiezza è dettata esclusivamente dalla feature del rispettivo nodo GNN.
            for i in range(num_atoms):
                if local_channel_name is not None:
                    # spostiamo il target del canale laser sul k-esimo atomo
                    seq.target(f"atom_{i}", 'local_rydberg')

                # Se features_np è un tensore PyTorch o un array multidimensionale, appiattiamolo
                if isinstance(features_np, torch.Tensor):
                    feat_flat = features_np[i].detach().cpu().numpy().flatten()
                else:
                    feat_flat = np.asarray(features_np[i]).flatten()
    
                # NOTA :
                # se il canale è globale non possiamo cambiare target,
                # per utilizzare le features di tutti gli atomi 
                # applichiamo una sequenza di impulsi per ciascun atomo, 
                # in slot temporali consecutivi (Time-Multiplexing), 
                # di modo che il sistema quantistico riceva comunque
                # le informazioni (senza operazioni distruttive come somma/media etc)
                
                # se il qubit è mappato da nodo/atomo di padding
                if mask_np[i]:
                    # valore unico per questo atomo specifico derivato dalla GNN
                    # Estraiamo correttamente il valore scalare dall'array/tensore i-esimo
                    val_scalar = float(feat_flat[i] if i < len(feat_flat) else feat_flat[0])
                    safe_amp = np.clip(val_scalar, min_amplitude, max_amplitude)
                else:
                    # altrimenti spegniamo il laser (ampiezza = 0.0) così non disturba la simulazione
                    safe_amp = 0.0  

                # durata sicura per atomo 1000 ns totali / num_atoms
                # la durata deve essere un multiplo del clock period
                # vedi con self.device.channels['rydberg_global'].clock_period
                slot_duration = max(16, (1000 // num_atoms) // 4 * 4)  # Minimo 16 ns, multiplo di 4
                
                # impulso usando ConstantPulse 
                pulse = Pulse.ConstantPulse(
                    duration=slot_duration,             # durata in ns (multiplo del clock period)
                    amplitude=safe_amp,        # ampiezza guidata dalla GNN per atomo
                    detuning=0.0,
                    phase=0.0
                )
                if local_channel_name is not None: 
                    seq.add(pulse, 'local_rydberg')
    
                else: 
                    seq.add(pulse, 'rydberg_global')
            return seq

        def simulate_single(i):
            # ==== costruzione sequenza parametrizzata per il campione corrente =======
            seq = _build_sequence(x_projected[i], mask[i]) # singoli campioni del batch

            #TODO: Emulatori basati su Tensor Networks per poter fare training con più qubit su simulatore?? 
            if backend_name == 'neutral_atom_sim':
                if noise_model is not None:
                    config = QutipConfig(
                        noise_model=noise_model,    #None se non è prevista noise
                        prefer_device_noise_model=(noise_model is not None),
                        observables=[BitStrings(num_shots=50)],
                        options={"nsteps": 50000}  # limite MASSIMO di step
                    )
                else:
                    config = QutipConfig(
                        prefer_device_noise_model=(noise_model is not None),
                        observables=[BitStrings(num_shots=50)],
                        options={"nsteps": 50000}  # limite MASSIMO di step
                    )
                # Istanziamo il backend locale con la sequenza e la configurazione opzionale
                backend = QutipBackendV2(sequence=seq, config=config, mimic_qpu=True)
                results = backend.run()
                logger.debug(f"results: {results}")
                # misura finale e campionamento delle stringhe di bit
                string_counts = results.get_result("bitstrings", 1.0)
                total_shots = sum(string_counts.values()) if string_counts else 1

                # init vettore di probabilità qubits
                qubit_probs = np.zeros(num_atoms, dtype=np.float32)
                if total_shots > 0 and string_counts:
                    #per ogni bitstring
                    for bitstring, count in string_counts.items():
                        for atom_idx in range(num_atoms):
                            # se lo stato del qubit è '1' aggiungi le occorrenze misurate
                            if atom_idx < len(bitstring) and bitstring[atom_idx] == '1':
                                qubit_probs[atom_idx] += count
                    
                    # normalizzazione per numero shots
                    qubit_probs /= float(total_shots)
                
                return qubit_probs.tolist() # [num_atoms]
            else:
                raise NotImplementedError("Real QPU training not implemented")

        # Esecuzione parallela sul batch tramite ThreadPool
        try:
            outputs = list(executor.map(simulate_single, range(batch_size)))
        except Exception as e:
            logger.error(f"unespected error {e}")
            raise RuntimeError(f"errore inaspettato nel forward pass del layer di classificazione pulser: {e}")

        logger.info(f"Valori di probabilità atomica nel batch: min={min([min(o) for o in outputs])}, max={max([max(o) for o in outputs])}, mean={np.mean(outputs)}")
        # converti lista di output in tensore PyTorch [batch_size, num_atoms]
        out_tensor = torch.tensor(outputs, device=x_projected.device, dtype=torch.float32)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # STE (Straight-Through Estimator) adattato alla riduzione dimensionale
        x_projected, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(-1).expand_as(x_projected)
        return grad_input, None, None, None, None, None, None