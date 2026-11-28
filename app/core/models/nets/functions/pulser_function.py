import torch
import torch.nn as nn
import numpy as np
import logging
import concurrent.futures
from functools import partial
from pulser import Register, Sequence, Pulse
from pulser.backend import NoiseModel, StateResult
from pulser.backend import BitStrings
from emu_sv import SVBackend, SVConfig
from pulser.register import RectangularLatticeLayout
from pulser.waveforms import CustomWaveform

from app.core.utils import backend
from app.core.utils.parallel_utils import ExecutorSingleton

logger = logging.getLogger("PulserFunction")


class PulserFunction():#torch.autograd.Function):
    log_printed = False

    @staticmethod
    def _build_register(num_qubits, qubit_max_origin_distance, qubit_min_distance, max_layout_fill_factor, qubit_positions_coords):
        # limite distanza massima dall'origine
        if isinstance(qubit_positions_coords, torch.Tensor):
            coords_t = torch.clamp(qubit_positions_coords, min=-qubit_max_origin_distance, max=qubit_max_origin_distance)
            coords_np = coords_t.detach().cpu().numpy()
        else:
            coords_np = np.clip(qubit_positions_coords, -qubit_max_origin_distance, qubit_max_origin_distance)

    
        # rispetto del fill factor necessario 
        total_traps_needed = num_qubits / max_layout_fill_factor
        # limite distanza minima tra qubits e
        cols = max(4, int(np.ceil(np.sqrt(total_traps_needed))))
        rows = int(np.ceil((total_traps_needed + cols - 1) // cols))
        spacing = max(6.0, qubit_min_distance + 1.0)

        # analog device richiede layout 
        layout = RectangularLatticeLayout(rows=rows, columns=cols, col_spacing=spacing, row_spacing=spacing)
        layout_coords = layout.coords # Elenco delle coordinate di tutte le trappole disponibili nel layout

        # associamo ciascuna coordinata (x, y) della GAT alla trappola libera più vicina
        chosen_trap_indices = []
        used_traps = set()
        for q_idx in range(num_qubits):
            target_pos = coords_np[q_idx]
            best_trap_id = -1
            min_d = float('inf')
            
            for trap_id, trap_coord in enumerate(layout_coords):
                # se la trappola è già stata scelta, salta
                if trap_id in used_traps:
                    continue
                # norma euclidea tra target della GAT e trappola candidata
                d = np.linalg.norm(target_pos - trap_coord)
                if d < min_d:
                    min_d = d
                    best_trap_id = trap_id
            
            if best_trap_id != -1:
                chosen_trap_indices.append(best_trap_id)
                used_traps.add(best_trap_id)
            else:
                # fallback di sicurezza se la griglia è satura
                available = [i for i in range(len(layout_coords)) if i not in used_traps]
                chosen_trap_indices.append(available[0])
                used_traps.add(available[0])

        # creazione formale del Register a partire dal layout (Risolve l'errore di AnalogDevice)
        qubit_ids = [f"atom_{i}" for i in range(num_qubits)]
        register = layout.define_register(*chosen_trap_indices, qubit_ids=qubit_ids)
        return register
        
        
        
    @staticmethod
    def _build_sequence(atom_features_sample, atom_mask_sample, register, device, backend_params, waves_size, phase_param_size):
        """
        Costruisce la sequenza di impulsi Pulser mappando le feature 
        distinte di ciascun atomo della GNN sui rispettivi target del circuito.
        @param: 
            atom_features_sample: features di tutto il grafo: [num_nodes, params_per_node]
            atom_mask_sample: mask per specifica molecola: [num_nodes]
        """ 
        min_amplitude = backend_params.get("min_amplitude", 0.0)
        max_amplitude = backend_params.get("max_amplitude", 12.556)
        min_detuning = backend_params.get("min_detuning", -125.56)
        max_detuning = backend_params.get("max_detuning", 125.56)
        seq = Sequence(register, device)

        # verifica la disponibilità di un canale locale o usa il canale locale di Rydberg global
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

        # trasforma tensore pytorch in numpy forma sempre: [num_nodes, params_per_node]
        if isinstance(atom_features_sample, torch.Tensor):
            features_np = atom_features_sample.detach().cpu().numpy()
        else:
            features_np = np.asarray(atom_features_sample)
        if features_np.ndim == 1:
            # forza forma 2D [num_nodes, 1] se era 1D (per gestire diversi numeri di param)
            features_np = features_np[:, np.newaxis] 
        # maschera decisionale per applicare l'impulso laser su determinato qubit
        if isinstance(atom_mask_sample, torch.Tensor):
            mask_np = atom_mask_sample.detach().cpu().numpy() # [max_num_nodes] di booleani con num_nodes elementi a true
        else:
            mask_np = np.asarray(atom_mask_sample) 
        num_nodes = int(mask_np.sum().item())
        
        # iteriamo i nodi della gat e applichiamo per ognuno un impulso 
        # la cui ampiezza è dettata esclusivamente dalla feature del rispettivo nodo GNN.
        for i in range(num_nodes):
            pulse = PulserFunction._build_pulse(
                i,
                seq,
                mask_el=mask_np[i], 
                features_el=features_np[i], 
                waves_size=waves_size,
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude,
                min_detuning=min_detuning,
                max_detuning=max_detuning,
                local_channel_name=local_channel_name, 
                phase_param_size=phase_param_size, 
                num_nodes=num_nodes
            )
            
            if local_channel_name is not None: 
                seq.add(pulse, 'local_rydberg')

            else: 
                seq.add(pulse, 'rydberg_global')
        return seq

    @staticmethod
    def _build_pulse(target_qubit: int, seq: Sequence, mask_el, features_el, waves_size, min_amplitude, max_amplitude, min_detuning, max_detuning, local_channel_name, phase_param_size, num_nodes):
        safe_amp, safe_det, safe_phase = None, None, None
        amp_waveform, det_waveform = None, None
        is_real_node = bool(mask_el)

        if is_real_node:
            # parametri nodo i-esimo
            feat_node = features_el

            # meno di 3 parametri 
            if waves_size <= 1:
                amp_scalar = float(feat_node[0]) if feat_node.size > 0 else 0.0
                det_scalar = float(feat_node[1]) if feat_node.size > 1 else 0.0
                phase_scalar = float(feat_node[2]) if feat_node.size > 2 else 0.0
                safe_amp = np.clip(amp_scalar, min_amplitude, max_amplitude)
                safe_det = np.clip(det_scalar, min_detuning, max_detuning)
                safe_phase = np.clip(phase_scalar, -6.28, 6.28) #radianti
            elif waves_size % 4 == 0:
                raw_amp_values = feat_node[:waves_size] # primi N valori
                raw_det_values = feat_node[waves_size : 2 * waves_size] # N valori in mezzo
                raw_phase_values = feat_node[2*waves_size: 2*waves_size+phase_param_size] # ultimo valore phase
                amp_values = np.clip(raw_amp_values, min_amplitude, max_amplitude) 
                det_values = np.clip(raw_det_values, min_detuning, max_detuning)
                phase_values = np.clip(raw_phase_values, -6.28, 6.28)
                amp_waveform = CustomWaveform(amp_values)
                det_waveform = CustomWaveform(det_values)
                safe_phase = float(phase_values[0] if phase_values.size > 0 else 0.0)
            else: 
                raise ValueError("Numero di parametri deve essere minore di 3 o divisibile per 4")
        else:
            # altrimenti spegniamo il laser (ampiezza = 0.0) così non disturba la simulazione
            if waves_size <= 1:
                safe_amp = 0.0
                safe_det = 0.0
                safe_phase = 0.0
            elif waves_size % 4 == 0:
                amp_values = np.zeros(waves_size)
                det_values = np.zeros(waves_size)
                amp_waveform = CustomWaveform(amp_values)
                det_waveform = CustomWaveform(det_values)
                safe_phase = 0.0
            else:
                raise ValueError("Numero di parametri deve essere minore di 3 o divisibile per 4")

        if local_channel_name is not None:
            # spostiamo il target del canale laser sul k-esimo atomo
            seq.target(f"atom_{target_qubit}", 'local_rydberg')

        if amp_waveform is None:
            slot_duration = max(16, (1000 // max(1, num_nodes)) // 4 * 4) # Minimo 16 ns, multiplo di 4
            pulse = Pulse.ConstantPulse(
                duration=slot_duration,
                amplitude=safe_amp,
                detuning=safe_det,
                phase=safe_phase
            )
        else: 
            pulse = Pulse(
                amplitude=amp_waveform,
                detuning=det_waveform,
                phase=safe_phase
            )
        return pulse

    @staticmethod
    def build_sv_config(backend_name, noise_model):
        if backend_name == 'neutral_atom_sim':
            # Usa la signature corretta per evitare AttributeError su samples_per_run
            if noise_model is not None:
                return SVConfig(
                    noise_model=noise_model,
                    prefer_device_noise_model=True,
                    observables=[StateResult()],
                    #observables=[BitStrings(num_shots=50)],
                    #options={"n_steps": 50000},
                    dt=10,  # Passo di discretizzazione in ns (default è 10 ns; abbassa a 4 o 5 se hai impulsi molto rapidi, alza a 20 per velocizzare)
                    precision="complex64",  # Usa complex64 su T4/P100 per dimezzare la VRAM e raddoppiare la velocità rispetto a complex128
                )
            else:
                return SVConfig(
                    observables=[StateResult()],
                    #observables=[BitStrings(num_shots=50)],
                    #options={"n_steps": 50000},   
                    dt=10,  # Passo di discretizzazione in ns (default è 10 ns; abbassa a 4 o 5 se hai impulsi molto rapidi, alza a 20 per velocizzare)
                    precision="complex64",  # Usa complex64 su T4/P100 per dimezzare la VRAM e raddoppiare la velocità rispetto a complex128          
                )
        else:
            raise NotImplementedError("Real QPU training not implemented")

    @staticmethod
    def simulate_single(x_projected_sample, mask_sample, device, backend_params, waves_size, phase_param_size, 
                        backend_name, noise_model, num_qubits, qubit_positions_embedding_sample, 
                        qubit_max_origin_distance, qubit_min_distance, max_layout_fill_factor):
        
        register = PulserFunction._build_register(
            num_qubits, qubit_max_origin_distance, qubit_min_distance, max_layout_fill_factor, qubit_positions_embedding_sample
        )
        # ==== costruzione sequenza parametrizzata per il campione corrente =======
        seq = PulserFunction._build_sequence(
            x_projected_sample, 
            mask_sample, 
            register,
            device, 
            backend_params, 
            waves_size, 
            phase_param_size
        )

        config = PulserFunction.build_sv_config(backend_name, noise_model)
        backend = SVBackend(sequence=seq, config=config, mimic_qpu=True)
        results = backend.run()
        logger.debug(f"results: {results}")
        if hasattr(results, "get_final_state"):
            result = results.get_final_state()
        else:
            result = results.get_result("state", 1.0)

        return PulserFunction.extract_statevector_features(result, num_qubits)

    @staticmethod
    def extract_single_result(string_counts, num_qubits):
        total_shots = sum(string_counts.values()) if string_counts else 1
        
        qubit_probs = np.zeros(num_qubits, dtype=np.float32)
        if total_shots > 0 and string_counts:
            for bitstring, count in string_counts.items():
                for atom_idx in range(num_qubits):
                    if atom_idx < len(bitstring) and bitstring[atom_idx] == '1':
                        qubit_probs[atom_idx] += count
            qubit_probs /= float(total_shots)
        logger.debug(f"[DEBUG SIM] Risultato simulazione batch (mean): {np.mean(qubit_probs):.4f}")
        return qubit_probs.tolist()

    @staticmethod
    def forward(#ctx, 
                x_projected, mask, device, backend_name, use_noise, backend_params, waves_size, phase_param_size, qubit_positions):
        """
        ctx.device_backend = backend_name
        ctx.use_noise = use_noise
        ctx.backend_params = backend_params
        ctx.waves_size = waves_size
        ctx.phase_param_size = phase_param_size
        ctx.pulser_device = device
        """

        executor = ExecutorSingleton.get_executor(max_workers=8)
        if torch.isnan(x_projected).any() or torch.isinf(x_projected).any():
            logger.warning("ATTENZIONE: Trovati NaN o Inf nelle feature proiettate dalla GNN!")
        
        batch_size = x_projected.shape[0]
        num_qubits = backend_params["max_qubits"]
        qubit_max_origin_distance = backend_params.get("qubit_max_origin_distance", 38)
        qubit_min_distance = backend_params.get("qubit_min_distance", 5)
        max_layout_fill_factor = backend_params.get("max_layout_fill_factor", 0.5)
        
        noise_model = None
        if backend_name == 'neutral_atom_sim' and use_noise:
            noise_model = NoiseModel(
                state_prep_error=0.01,
                dephasing_rate=0.05,
                temperature=10.0,
                trap_waist=1.0,
                trap_depth=100.0
            )

        classical_device = backend.get_optimal_device()
        is_gpu = classical_device.type in ["cuda", "mps"]

        # Preparazione worker forward indicizzando correttamente il singolo campione [b]
        x_proj_np = x_projected.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        qubit_pos_np = qubit_positions.detach().cpu().numpy()
        if not is_gpu:
            logger.info(f"--- [FORWARD QPU/SIM] Esecuzione parallela su CPU (max_workers={executor._max_workers}) ---")
            outputs = PulserFunction.handle_cpu_forward(
                batch_size, x_proj_np, mask_np, qubit_pos_np, device, backend_params, waves_size, phase_param_size, backend_name, noise_model, num_qubits, qubit_max_origin_distance, qubit_min_distance, max_layout_fill_factor, executor
            )
        else:
            logger.info(f"--- [FORWARD QPU/SIM] Esecuzione sequenziale su GPU (device={classical_device}) ---")
            outputs = []
            for b in range(batch_size):
                res = PulserFunction.simulate_single(
                    x_projected_sample=x_proj_np[b],
                    mask_sample=mask_np[b],
                    device=device,
                    backend_params=backend_params,
                    waves_size=waves_size,
                    phase_param_size=phase_param_size,
                    backend_name=backend_name,
                    noise_model=noise_model,
                    num_qubits=num_qubits,
                    qubit_positions_embedding_sample=qubit_pos_np[b],
                    qubit_max_origin_distance=qubit_max_origin_distance,
                    qubit_min_distance=qubit_min_distance,
                    max_layout_fill_factor=max_layout_fill_factor
                )
                outputs.append(res)
                logger.debug(f"[FORWARD] Completata molecola {b+1}/{batch_size} | mean_prob: {np.mean(res):.3f}")
        out_tensor = torch.tensor(np.array(outputs), device=x_projected.device, dtype=torch.float32)
        #ctx.save_for_backward(x_projected, mask, qubit_positions, out_tensor)
        return out_tensor

    def handle_cpu_forward(batch_size, x_proj_np, mask_np, qubit_pos_np, device, backend_params, waves_size, phase_param_size, backend_name, noise_model, num_qubits, qubit_max_origin_distance, qubit_min_distance, max_layout_fill_factor, executor):
        tasks = []
        for b in range(batch_size):
            simulate_fn = partial(
                PulserFunction.simulate_single,
                x_projected_sample=x_proj_np[b],           # Singolo campione 2D: [max_nodes, params_per_node]
                mask_sample=mask_np[b],                     # Singola maschera 1D: [max_nodes]
                device=device,
                backend_params=backend_params,
                waves_size=waves_size,
                phase_param_size=phase_param_size,
                backend_name=backend_name,
                noise_model=noise_model,
                num_qubits=num_qubits,
                qubit_positions_embedding_sample=qubit_pos_np[b], # Singolo campione coordinate 2D: [num_qubits, 2]
                qubit_max_origin_distance=qubit_max_origin_distance,
                qubit_min_distance=qubit_min_distance,
                max_layout_fill_factor=max_layout_fill_factor
            )
            tasks.append(simulate_fn)

        # Esecuzione parallela sul batch tramite ThreadPool
        future_to_idx = {executor.submit(fn): idx for idx, fn in enumerate(tasks)}
        outputs = [None] * batch_size
        completed = 0

        logger.info(f"--- [FORWARD QPU/SIM] Inizio simulazione Batch (Totale: {batch_size} molecole) ---")
        try:
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                outputs[idx] = res
                completed += 1
                
                # Logga l'avanzamento (es. ogni 4 molecole o ad ogni singola molecola)
                if completed == batch_size:
                    logger.info(f"[FORWARD] Completate {completed}/{batch_size} molecole ({(completed/batch_size)*100:.1f}%) | Ultimo mean_prob: {np.mean(res):.3f}")
        except Exception as e:
            logger.error(f"unexpected error: {e}")
            raise RuntimeError(f"errore inaspettato nel forward pass del layer di classificazione pulser:  {e}")
        return outputs

    
    @staticmethod
    def extract_statevector_features(final_state, num_qubits):
        """
        Estrae il reservoir quantistico: sia le probabilità a singolo corpo <n_i>
        sia le correlazioni a due corpi <n_i n_j> (entanglement / Rydberg blockade footprint).
        """
        probs_dict = final_state.probabilities() if hasattr(final_state, "probabilities") else {}
        
        #  dizionario standard delle probabilità degli stati di base
        single_probs = np.zeros(num_qubits, dtype=np.float32)
        two_body = np.zeros((num_qubits, num_qubits), dtype=np.float32)
        
        for state_str, p in probs_dict.items():
            if p <= 1e-7:
                continue
            clean_str = state_str.replace("|", "").replace(">", "").replace(",", "").strip()
            bits = [1 if c in ('1', 'r', 'R') else 0 for c in clean_str[:num_qubits]]
            
            # 1-corpo
            for i in range(num_qubits):
                if bits[i] == 1:
                    single_probs[i] += p
                    
            # 2-corpi
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    if bits[i] == 1 and bits[j] == 1:
                        two_body[i, j] += p

        # Estraiamo la parte triangolare superiore delle coppie (15 valori per 6 qubit)
        upper_pairs = []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                # Covarianza quantistica connessa: <n_i n_j> - <n_i><n_j>
                cov = two_body[i, j] - (single_probs[i] * single_probs[j])
                upper_pairs.append(cov)
                
        #  6 valori 1-corpo + 15 correlatori = 21 feature
        features = np.concatenate([single_probs, np.array(upper_pairs, dtype=np.float32)])
        return features.tolist()

    def compute_quantum_fidelity(state_A, state_B):
        """
        Calcola la fedeltà quantistica K(A, B) = |<psi_A | psi_B>|^2
        """
        psi_A = state_A.state_vector.full().flatten() # vettore numpy [64]
        psi_B = state_B.state_vector.full().flatten()
        
        # Prodotto scalare Hermitiano
        inner_prod = np.vdot(psi_A, psi_B)
        fidelity = np.abs(inner_prod) ** 2
        return float(fidelity)