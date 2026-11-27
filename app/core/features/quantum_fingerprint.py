import os
import hashlib
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pulser import Pulse
from emu_sv import SVBackend, SVConfig
import torch
from rdkit import Chem

from pulser.devices import AnalogDevice
from pulser.register import RectangularLatticeLayout
from pulser.sequence import Sequence
from pulser.waveforms import BlackmanWaveform, RampWaveform
from pulser.backend import StateResult
from pulser_simulation.simresults import SimulationResults
from app.core.configs.configger import PipelineConfigger
from app.core.utils.backend import get_optimal_device
logger = logging.getLogger(__name__)

configger = PipelineConfigger()

def _canonicalize_smiles(smiles: str) -> str:
    """Restituisce il canonical SMILES per garantire chiavi di cache univoche."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def _simulate_single_smiles_reservoir(smi: str, params: dict) -> list[float]:
    """
    esegue la simulazione Pulser su una singola molecola ed estrae 
    le feature del Quantum Reservoir leggendo i limiti fisici dal dizionario params.
    """
    num_qubits = params.get("max_qubits", 6)
    expected_dim = num_qubits + (num_qubits * (num_qubits - 1)) // 2

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [0.0] * expected_dim

    # lettura parametri
    qubit_min_distance = params.get("qubit_min_distance", 5.0)
    spacing = max(6.0, float(qubit_min_distance) + 1.0)

    # layout che rispetta fill factor 
    total_traps = num_qubits / p["backend_name"].get("max_layout_fill_factor", 0.5)
    cols = max(4, int(np.ceil(np.sqrt(total_traps))))
    rows = int(np.ceil((total_traps + cols - 1) // cols))
    layout = RectangularLatticeLayout(rows=rows, columns=cols, col_spacing=spacing, row_spacing=spacing)
    
    qubit_ids = [f"atom_{i}" for i in range(num_qubits)]
    register = layout.define_register(*list(range(num_qubits)), qubit_ids=qubit_ids)

    # canale rydberg globale per tutti i qubit
    seq = Sequence(register, AnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    channel_obj = AnalogDevice.channels["rydberg_global"]

    # durata impulso standard per transizione adiabatica
    pulse_duration = 1600  # ns (1.6 us)
    duration_us = pulse_duration * 1e-3

    # ampiezza (rad/µs)
    max_amp_cfg = float(params.get("max_amplitude", 12.566))
    max_amp_hardware = channel_obj.max_amp if channel_obj.max_amp is not None else 12.566
    # margine di sicurezza 15%
    target_peak_amp = min(max_amp_cfg, max_amp_hardware) * 0.85  

    # BlackmanWaveform: area della campana è il 42% dell'area totale
    # Area = Peak * 0.42 * T_us
    safe_area = float(target_peak_amp * 0.42 * duration_us)
    amp_wf = BlackmanWaveform(pulse_duration, area=safe_area)

    # limiti Detuning
    min_det_cfg = float(params.get("min_detuning_feat", -6.28))
    max_det_cfg = float(params.get("max_detuning_feat", 6.28))

    # controlla se i limiti vanno bene per il canale
    if channel_obj.max_abs_detuning is not None:
        hw_max_det = float(channel_obj.max_abs_detuning)
        start_det = max(min_det_cfg, -hw_max_det)
        stop_det = min(max_det_cfg, hw_max_det)
    else:
        start_det = min_det_cfg
        stop_det = max_det_cfg

    # waveform di detuning dal min al max in modo lineare
    det_wf = RampWaveform(pulse_duration, start=start_det, stop=stop_det)
    seq.add(Pulse(amp_wf, det_wf, phase=0.0), "rydberg_global")
    is_gpu = get_optimal_device().type in ["cuda", "mps"]
    # esecuzione con StateResult
    config = SVConfig(
        observables=[StateResult()],
        options={"nsteps": 25000}
    )
    backend = SVBackend(sequence=seq, config=config, mimic_qpu=True)
    results = backend.run()

    if hasattr(results, "get_final_state"):
        final_state = results.get_final_state()
    else:
        final_state = results.get_result("state", 1.0)

    # estrazione 1-body e 2-body correlators (Reservoir)
    single_probs = np.zeros(num_qubits, dtype=np.float32)
    two_body = np.zeros((num_qubits, num_qubits), dtype=np.float32)

    if hasattr(final_state, "probabilities"):
        probs_obj = final_state.probabilities()
        if isinstance(probs_obj, dict):
            for state_str, p in probs_obj.items():
                if p <= 1e-7:
                    continue
                #pulisci la stringa di stato e converti in bit
                clean = state_str.replace("|", "").replace(">", "").replace(",", "").strip()
                bits = [1 if c in ('1', 'r', 'R') else 0 for c in clean[:num_qubits]]
                
                for i in range(num_qubits):
                    if bits[i] == 1:
                        single_probs[i] += p
                
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        if bits[i] == 1 and bits[j] == 1:
                            two_body[i, j] += p

    # covarianza quantistica: C_ij = <n_i n_j> - <n_i><n_j>
    # misuro quanto lo stato di due atomi varia in modo dipendente o indipendente
    upper_pairs = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            cov = two_body[i, j] - (single_probs[i] * single_probs[j])
            upper_pairs.append(cov)

    
    features = np.concatenate([single_probs, np.array(upper_pairs, dtype=np.float32)])
    return features.tolist()


def compute_quantum_fingerprints(
    smiles_list: list[str], 
    params: dict = None, 
    cache_dir: Path = configger.global_cfg.get("quantum_cache_dir", ".cache/quantum_fingerprints"),
    max_workers: int = 4
) -> list[list[float]]:
    """
    calcola o carica dalla cache le Quantum Fingerprint (21 feature) per una lista di SMILES,
    garantendo coerenza con i vincoli passati in params.
    """
    if params is None:
        params = {}

    num_qubits = params.get("max_qubits", 6)
    expected_dim = num_qubits + (num_qubits * (num_qubits - 1)) // 2

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Hash univoco dei parametri critici di configurazione
    hash_keys = {
        "max_qubits": num_qubits,
        "max_amp": params.get("max_amplitude", 12.566),
        "min_det": params.get("min_detuning_feat", -6.28),
        "max_det": params.get("max_detuning_feat", 6.28),
        "min_dist": params.get("qubit_min_distance", 5),
    }
    config_hash = hashlib.md5(json.dumps(hash_keys, sort_keys=True).encode()).hexdigest()[:8]
    cache_file = cache_dir / f"quantum_fingerprints_{config_hash}.pt"

    cache_dict: dict[str, list[float]] = {}
    if cache_file.exists():
        try:
            cache_dict = torch.load(cache_file, map_location="cpu", weights_only=False)
            logger.info(f"[QUANTUM CACHE] Caricati {len(cache_dict)} fingerprint da {cache_file}")
        except Exception as e:
            logger.warning(f"[QUANTUM CACHE] Impossibile leggere il file di cache: {e}. Creazione nuovo dizionario.")

    canonical_smiles = [_canonicalize_smiles(s) for s in smiles_list]
    missing_smiles = list({s for s in canonical_smiles if s not in cache_dict})

    if missing_smiles:
        logger.info(
            f"[QUANTUM FINGERPRINT] Mancano {len(missing_smiles)} molecole nella cache. "
            f"Avvio simulazioni parallele (workers={max_workers})..."
        )
        
        computed_count = 0
        #esecuzione parallela
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_smi = {
                executor.submit(_simulate_single_smiles_reservoir, smi, params): smi 
                for smi in missing_smiles
            }
            
            for future in as_completed(future_to_smi):
                smi = future_to_smi[future]
                try:
                    feat = future.result()
                    cache_dict[smi] = feat
                    computed_count += 1
                    
                    if computed_count % 16 == 0 or computed_count == len(missing_smiles):
                        logger.info(
                            f"[QUANTUM FINGERPRINT] Calcolate {computed_count}/{len(missing_smiles)} "
                            f"({(computed_count / len(missing_smiles)) * 100:.1f}%)"
                        )
                except Exception as e:
                    logger.error(f"[QUANTUM FINGERPRINT] Errore sulla molecola {smi}: {e}")
                    cache_dict[smi] = [0.0] * expected_dim

        # salvataggio atomico della cache aggiornata
        try:
            temp_cache_file = cache_dir / f"{cache_file.name}.tmp"
            torch.save(cache_dict, temp_cache_file)
            temp_cache_file.replace(cache_file)
            logger.info(f"[QUANTUM CACHE] Cache salvata in {cache_file} (Totale memorizzate: {len(cache_dict)})")
        except Exception as e:
            logger.error(f"[QUANTUM CACHE] Errore durante il salvataggio della cache: {e}")

    results = [cache_dict.get(s, [0.0] * expected_dim) for s in canonical_smiles]
    return results