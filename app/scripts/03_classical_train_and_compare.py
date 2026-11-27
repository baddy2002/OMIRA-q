# app/scripts/03_classical_train_and_compare.py
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys
import random
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score


from app.core.configs.configger import PipelineConfigger
from app.core.features.quantum_fingerprint import compute_quantum_fingerprints
from app.core.models.nets.surrogate_pulser import SurrogatePulserNet
from app.core.models.xgb_estimator import XGBEstimator
from app.core.models.neural_estimator import NeuralEstimator
from app.core.models.nets.gatv2 import GATv2Net
from app.core.models.nets.egnn3 import EGNNNet
from app.core.models.nets.dmpnn import DMPNNNet
from app.core.utils.augmentation_utils import AugmentedPyGDataset

configger = PipelineConfigger()

# Configurazione del logging (Console + File di Debug in data/debug/training.log)
debug_dir = Path("data/debug")
debug_dir.mkdir(parents=True, exist_ok=True)
log_file_path = debug_dir / "training.log"

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler, file_handler],
)

logger = logging.getLogger("Classical_Train_and_Compare")

def load_dataset_views(parquet_path: Path):
    """
    carica il parquet preprocessato, 
    legge i task ufficiali e li stampa per tracciabilità
    restituisce i dati divisi per split per evitare data leakage.
    """
    df = pd.read_parquet(parquet_path)
    logger.info(f"Dataset caricato da {parquet_path}. Totale righe: {len(df)}")
    
    task_cols = [c for c in df.columns if c.startswith("y_")]
    logger.info(f"Task biologici rilevati ({len(task_cols)}): {task_cols}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df, task_cols

def compute_morgan_fingerprints(smiles_list: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """
    Calcola i Morgan Fingerprints usando il generatore di RDKit.
    """
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # estrae direttamente l'array numpy dei bit
            arr_np = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(arr_np)
        else:
            fps.append(np.zeros((n_bits,), dtype=np.float32))
    return np.array(fps)

from rdkit.Chem import rdchem

# quando due atomi si legano formando una molecola gli orbitali atomici 
# puri (s,p,d ed f) si mescolano dando vita a questi orbitali ibridi:
_HYBRID_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

def featurize_atom(atom) -> list[float]:
    """
    Feature di nodo continue (NUM_NODE_FEATURES = 12).

    Ordine:
      1  numero atomico (normalizzato /100)
      2  grado (numero di vicini heavy-atom)
      3  idrogeni totali implicit+explicit (/4)
      4  carica formale dell'atomo
      5  radicali elettronici (/2)  (elettrone spaiato, cioè non accoppiato in un legame o in un doppietto elettronico)
      6  se la molecola è aromatica e ha elettroni delocalizzati (0/1)
      7  se la molecola è un anello (0/1)
      8-12 ibridazione one-hot: SP, SP2, SP3, SP3D, SP3D2 (geometria 3D degli orbitali leganti attorno all'atomo)
    """
    hyb = atom.GetHybridization()
    hyb_onehot = [1.0 if hyb == h else 0.0 for h in _HYBRID_TYPES]
    return [
        atom.GetAtomicNum() / 100.0,  
        float(atom.GetDegree()),        
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetFormalCharge()),
        atom.GetNumRadicalElectrons() / 2.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *hyb_onehot,
    ]

# tipi di legami tra i due atomi
_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

# tabella approssimativa delle elettronegatività di Pauling per i principali atomi organici
_PAULING_EN = {
    1: 2.20,   # H
    2: 0.0,    # He
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    10: 0.0,   # Ne
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    18: 0.0,   # Ar
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    31: 1.81,  # Ga
    32: 2.01,  # Ge
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    36: 3.00,  # Kr
    37: 0.82,  # Rb
    38: 0.95,  # Sr
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.6,   # Nb
    42: 2.16,  # Mo
    43: 1.9,   # Tc
    44: 2.2,   # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    49: 1.78,  # In
    50: 1.96,  # Sn
    51: 2.05,  # Sb
    52: 2.1,   # Te
    53: 2.66,  # I
    54: 2.6,   # Xe
    55: 0.79,  # Cs
    56: 0.89,  # Ba
    57: 1.10,  # La
    58: 1.12,  # Ce
    59: 1.13,  # Pr
    60: 1.14,  # Nd
    61: 1.13,  # Pm
    62: 1.17,  # Sm
    63: 1.2,   # Eu
    64: 1.20,  # Gd
    65: 1.2,   # Tb
    66: 1.22,  # Dy
    67: 1.23,  # Ho
    68: 1.24,  # Er
    69: 1.25,  # Tm
    70: 1.1,   # Yb
    71: 1.27,  # Lu
    72: 1.3,   # Hf
    73: 1.5,   # Ta
    74: 2.36,  # W
    75: 1.9,   # Re
    76: 2.2,   # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54,  # Au
    80: 2.00,  # Hg
    81: 1.62,  # Tl
    82: 2.33,  # Pb
    83: 2.02,  # Bi
    84: 2.0,   # Po
    85: 2.2,   # At
    86: 2.2,   # Rn
    87: 0.7,   # Fr
    88: 0.9,   # Ra
    89: 1.1,   # Ac
    90: 1.3,   # Th
    91: 1.5,   # Pa
    92: 1.38,  # U
    93: 1.36,  # Np
    94: 1.28,  # Pu
    95: 1.3,   # Am
    96: 1.3,   # Cm
    97: 1.3,   # Bk
    98: 1.3,   # Cf
    99: 1.3,   # Es
    100: 1.3,  # Fm
    101: 1.3,  # Md
    102: 1.3,  # No
    103: 1.3,  # Lr
}

def featurize_bond(bond) -> list[float]:
    """
    feature di arco (NUM_EDGE_FEATURES = 7).

    Ordine:
      1-4  tipo di legame one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
      5    coniugato (0/1) (tre o più orbitali p adiacenti sono paralleli e in grado di sovrapporsi parzialmente lungo una catena)
      6    se è in un anello (0/1)
      7
        OLD    legame con H implicito o esplicito (0/1)  
        NOW    differenza di elettronegatività (scala di Pauling) tra i due atomi (Dice al modello quanto un legame è polarizzato)
    """
    bt = bond.GetBondType()
    bond_onehot = [1.0 if bt == b else 0.0 for b in _BOND_TYPES]
    # differenza di elettronegatività tra i due atomi
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    en1 = _PAULING_EN.get(atom1.GetAtomicNum(), 2.5)  # default a C se non trovato
    en2 = _PAULING_EN.get(atom2.GetAtomicNum(), 2.5)
    en_diff = abs(en1 - en2)

    # normalizzazione per delta massimo teorico ( 4.0 - 0.7 = 3.3)
    en_diff_normalized = min(en_diff / 3.3, 1.0)
    return [
        *bond_onehot,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        #0.0,   
       en_diff_normalized,  
    ]

def df_to_pyg_data_list(df: pd.DataFrame, task_cols: list[str], model_name: str = "gatv2", backend_name: str = "neutral_atom_sim") -> list[Data]:
    """
    converte il DataFrame in oggetti PyG Data con feature di nodo, arco, coordinate 3D
    e supporto opzionale per le Fingerprint di Morgan/quantistiche e Data Augmentation 3D (SO(3)) mirata per EGAT.
    """
    from rdkit import Chem

    data_list = []
    n_random_pos = 0
    
    gcfg = configger.get_model_config(model_name)
    p: dict = gcfg.get("params", {})
    if backend_name != "classical":
        bcfg = configger.get_backend_config(backend_name)
        pb = bcfg.get("params", {})
        p.update(pb)  # unisce i parametri del modello e del backend
    # recupero dei parametri
    expected_node_f = p.get("num_node_features", 12)
    expected_edge_f = p.get("num_edge_features", 7)
    
    use_fp = p.get("use_morgan_fingerprint", gcfg.get("use_morgan_fingerprint", False))
    n_bits = p.get("fingerprint_n_bits", gcfg.get("fingerprint_n_bits", 16))
    radius = p.get("fingerprint_radius", gcfg.get("fingerprint_radius", 2))

    
    # morgan fingerprints
    fps = None
    if use_fp:
        fps = compute_morgan_fingerprints(df["smiles"].tolist(), n_bits=n_bits, radius=radius)
    
    # --- QUANTUM FINGERPRINT ---
    use_q_fp = p.get("use_quantum_fingerprint", gcfg.get("use_quantum_fingerprint", False))
    q_fps = None
    if use_q_fp:
        # funzione che calcola offline (o carica da cache su disco) il vettore di feature per ogni molecola
        # passiamo l'intera lista di smiles o il df
        logger.info(f"Calcolo/Caricamento Quantum Fingerprint per {len(df)} molecole...")
        q_fps = compute_quantum_fingerprints(df["smiles"].tolist(), p)
    # ----------------------------------------------------------------

    for row_pos, (idx, row) in enumerate(df.iterrows()):
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # features di nodo con controllo di validità su dimensione
        x = torch.tensor([featurize_atom(a) for a in mol.GetAtoms()], dtype=torch.float)
        assert x.shape[1] == expected_node_f, f"Feature nodo attese {expected_node_f}, ottenute {x.shape[1]}"

        # features archi e attributi
        src, dst, edge_attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            f = featurize_bond(bond)
            src += [i, j]
            dst += [j, i]
            edge_attrs += [f, f]

        if len(src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, expected_edge_f), dtype=torch.float)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
        assert edge_attr.shape[1] == expected_edge_f, (
            f"Feature arco attese {expected_edge_f}, ottenute {edge_attr.shape[1]}"
        )

        # coordinate 3D
        n_atoms = mol.GetNumAtoms()
        xyz_raw = row.get("xyz", None)
        pos = None
        if xyz_raw is not None:
            try:
                xyz_arr = np.array(list(xyz_raw), dtype=np.float32)
                if xyz_arr.ndim == 2 and xyz_arr.shape == (n_atoms, 3):
                    pos = torch.tensor(xyz_arr, dtype=torch.float)
            except Exception:
                pass
        if pos is None:
            pos = torch.randn((n_atoms, 3), dtype=torch.float) * 0.01
            n_random_pos += 1

        # target multi-task
        y_vals = [row[t] for t in task_cols]
        y = torch.tensor([y_vals], dtype=torch.float)

        # oggetto Data di base da passare ai modelli
        data_kwargs = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "pos": pos, "y": y}
        # se la configurazione richiede la fingerprint, la includiamo nel batch PyG usando l'indice corretto del dataframe
        if use_fp and fps is not None:
            data_kwargs["fingerprint"] = torch.tensor(fps[idx], dtype=torch.float)

        # fingerprint quantum
        if use_q_fp and q_fps is not None:
            data_kwargs["quantum_fingerprint"] = torch.tensor(q_fps[row_pos], dtype=torch.float)
        
        data_list.append(Data(**data_kwargs))

    logger.info(f"df_to_pyg_data_list: {len(data_list)}/{len(df)} grafi costruiti (Fingerprint attive: {use_fp})")
    return data_list

def evaluate_multitask_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """Calcola le metriche multi-task (ROC-AUC media ignorando i NaN)."""
    roc_aucs = []
    for i in range(y_true.shape[1]):
        true_col = y_true[:, i]
        pred_col = y_pred_prob[:, i]
        valid = ~np.isnan(true_col)
        if valid.sum() > 10 and len(np.unique(true_col[valid])) > 1:
            try:
                score = roc_auc_score(true_col[valid], pred_col[valid])
                roc_aucs.append(score)
            except Exception as e:
                logger.error(f"Errore nel calcolo ROC-AUC per il task {i}: {e}")
                pass

    mean_auc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    return {"mean_roc_auc": mean_auc, "task_aucs": roc_aucs}

def per_task_auc(y_true: np.ndarray, y_pred: np.ndarray, task_names: list[str]) -> dict:
    """ROC-AUC per singolo task, None dove il task non è valutabile (pochi valori o una sola classe)."""
    out = {}
    for i, t in enumerate(task_names):
        valid = ~np.isnan(y_true[:, i])
        if valid.sum() > 10 and len(np.unique(y_true[valid, i])) > 1:
            out[t] = round(float(roc_auc_score(y_true[valid, i], y_pred[valid, i])), 4)
        else:
            out[t] = None
    return out

from sklearn.metrics import average_precision_score


def multitask_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_names: list[str],
    k: int = 50,
) -> dict:
    """
    metriche orientate alla selezione di molecole da testare, per task e mediate.

    - precision@k: frazione di attivi tra le k molecole con probabilità più alta
    - enrichment@k: precision@k diviso per la prevalenza del task nel set di valutazione
    - auprc: average precision (area sotto la curva precision-recall)
    - roc_auc: area sotto la curva ROC (invariata, per confronto con la letteratura)

    I task non valutabili (meno di k+1 valori validi o una sola classe) restituiscono None.
    """
    out = {"per_task": {}}
    p_at_k, ef_at_k, auprc, roc = [], [], [], []

    for i, t in enumerate(task_names):
        valid = ~np.isnan(y_true[:, i])
        yt = y_true[valid, i]
        yp = y_pred[valid, i]

        if len(yt) <= k or len(np.unique(yt)) < 2:
            out["per_task"][t] = None
            continue

        
        prevalence = float(yt.mean())
        # k molecole con punteggio di previsione più alto
        top_idx = np.argsort(-yp)[:k]
        # calcola la percentuale di attive reali presenti in quel grupp
        precision_k = float(yt[top_idx].mean())
        enrichment_k = precision_k / prevalence if prevalence > 0 else float("nan")
        ap = float(average_precision_score(yt, yp))
        auc = float(roc_auc_score(yt, yp))

        out["per_task"][t] = {
            "precision_at_k": round(precision_k, 4),
            "enrichment_at_k": round(enrichment_k, 4),
            "auprc": round(ap, 4),
            "roc_auc": round(auc, 4),
            "prevalence": round(prevalence, 4),
        }
        p_at_k.append(precision_k)
        ef_at_k.append(enrichment_k)
        auprc.append(ap)
        roc.append(auc)

    def _mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    out.update({
        "k": k,
        "mean_precision_at_k": _mean(p_at_k),
        "mean_enrichment_at_k": _mean(ef_at_k),
        "mean_auprc": _mean(auprc),
        "mean_roc_auc": _mean(roc),
    })
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Training di un modello con un seed")
    p.add_argument("--model", required=True, choices=["xgboost", "gatv2", "egnn3", "dmpnn"], help="modello da addestrare")
    p.add_argument("--backend", required=True, choices=["classical", "neutral_atom_sim", "neutral_atom_hw", "dwave_sim"], help="backend da usare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, nargs="+", default=[50, 100],
                   help="budget di molecole testabili per report, uno o più valori")
    return p.parse_args()


def setup_logging(model: str, seed: int, backend: str) -> None:
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_file = debug_dir / f"train_{backend}_{model}_seed{seed}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def ranking_report(y_true: np.ndarray, y_pred: np.ndarray, task_cols: list[str], ks: list[int]) -> dict:
    """Report completo di un run: AUC, AUPRC e metriche di ranking per ogni k."""
    report = {
        "roc_auc_mean": evaluate_multitask_predictions(y_true, y_pred)["mean_roc_auc"],
        "per_task_roc_auc": per_task_auc(y_true, y_pred, task_cols),
    }
    for k in ks:
        r = multitask_ranking_metrics(y_true, y_pred, task_cols, k=k)
        report[f"k{k}"] = {
            "precision_at_k": r["mean_precision_at_k"],
            "enrichment_at_k": r["mean_enrichment_at_k"],
            "auprc": r["mean_auprc"],
            "per_task": r["per_task"],
        }
    return report


def run_xgboost(train_df, val_df, test_df, task_cols, seed: int):
    xcfg = configger.get_model_config("xgboost")
    n_bits = xcfg["training"]["fingerprint_n_bits"]
    radius = xcfg["training"]["fingerprint_radius"]

    X_train = compute_morgan_fingerprints(train_df["smiles"].tolist(), n_bits, radius)
    X_val = compute_morgan_fingerprints(val_df["smiles"].tolist(), n_bits, radius)
    X_test = compute_morgan_fingerprints(test_df["smiles"].tolist(), n_bits, radius)

    xcfg["params"]["random_state"] = seed  
    est = XGBEstimator(xcfg)
    est.fit(
        (X_train, train_df[task_cols].to_numpy()),
        (X_val, val_df[task_cols].to_numpy()),
    )
    est.save(configger.paths["models_output_dir"] / f"xgboost_seed{seed}")
    return est.predict_proba(X_test), None

def load_surrogate_net(input_dim, bp, p):
    """Carica il surrogato da usare nel train in warmup e in bw per derivabilità"""
    surrogate_net = SurrogatePulserNet(input_dim=input_dim, num_qubits=bp['max_qubits'], hidden_dim=p["hidden_dim"])
    surrogate_weights_path = Path("data/output/models/surrogate_weights.pt")
    if surrogate_weights_path.exists():
        state_dict = torch.load(surrogate_weights_path)        
        surrogate_net.load_state_dict(state_dict)
        # Congeliamo i pesi del surrogato così non si muovono durante il training della GNN
        for param in surrogate_net.parameters():
            param.requires_grad = True

        # Mettilo in eval mode (così LayerNorm e Dropout non mutano lo stato)
        #surrogate_net.eval()
        logger.info("Modello surrogato caricato e congelato con successo nella GATv2!")
    else:
        logger.warning("ATTENZIONE: Pesi del surrogato non trovati! Ricordati di far girare lo script 05.")
    return surrogate_net

def define_target_neural_net(model_name: str = "gatv2", backend_name: str = "classical", dataset_name: str = 'nci') -> torch.nn.Module:
    """Definisce l'architettura della rete neurale target in base al nome del modello."""
    gcfg = configger.get_model_config(model_name)
    bcfg = configger.get_backend_config(backend_name)
    ncicfg = configger.get_data_config(dataset_name)
    p = gcfg["params"]
    bp = bcfg["params"]
    surrogate_net=None
    logger.info(f"Using backend type: {backend_name}")
    if backend_name != 'classical':
        # l'input del surrogato dipende dalla dimensione delle waveform non dei parametri con cui li calcola la gnn
        input_dim=(bp['waves_size']*2+bp['phase_param_size'])*ncicfg['preprocess']['chemistry']['n_heavy_max'] 
        # aggiungi le posizioni
        input_dim += bp['max_qubits']*2
        surrogate_net = load_surrogate_net(input_dim, bp, p)
    if model_name == "gatv2":
        return GATv2Net(
            num_node_features=p["num_node_features"],
            num_edge_features=p["num_edge_features"],
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            heads=p["heads"],
            num_tasks=p["num_tasks"],
            surrogate_net = surrogate_net,
            dropout=p["dropout"],
            use_morgan_fingerprint=p.get("use_morgan_fingerprint", False),
            fingerprint_n_bits=p.get("fingerprint_n_bits", 256),
            backend_config=bcfg,
            use_hybrid=False,
            use_quantum_fingerprint=p.get("use_quantum_fingerprint", False)
        )
    elif model_name == "egnn3":
        return EGNNNet(
            num_node_features=p["num_node_features"],
            num_edge_features=p["num_edge_features"],
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            num_tasks=p["num_tasks"],
            dropout=p["dropout"],
            use_morgan_fingerprint=p.get("use_morgan_fingerprint", False),
            fingerprint_n_bits=p.get("fingerprint_n_bits", 256),
            backend_config=bcfg
        )
    elif model_name == "dmpnn":
        return DMPNNNet(
            num_node_features=p["num_node_features"],
            num_edge_features=p["num_edge_features"],
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            heads=p["heads"],
            num_tasks=p["num_tasks"],
            dropout=p["dropout"],
            use_morgan_fingerprint=p.get("use_morgan_fingerprint", False),
            fingerprint_n_bits=p.get("fingerprint_n_bits", 256),
            backend_config=bcfg
        )
    else:
        raise ValueError(f"Modello neurale non supportato: {model_name}")

def run_neural(train_df, val_df, test_df, task_cols, seed: int, model_name: str, backend_name: str, dataset_name: str):
    gcfg = configger.get_model_config(model_name)
    p = gcfg["params"]
    batch_size = gcfg.get("training", {}).get("batch_size", 64)

    # costruzione delle liste PyG Data base
    train_data_list = df_to_pyg_data_list(train_df, task_cols, model_name, backend_name)
    val_data_list = df_to_pyg_data_list(val_df, task_cols, model_name, backend_name)
    test_data_list = df_to_pyg_data_list(test_df, task_cols, model_name, backend_name)

    if model_name == "egnn3":
        # num_aug_per_mol=1 raddoppia i dati: 11.308 originali + 11.308 ruotati/jitterati = 22.616 campioni
        train_dataset = AugmentedPyGDataset(train_data_list, num_aug_per_mol=1, jitter_std=0.005)
        logger.info(
            f"Oversampling 3D attivo per EGAT: Dataset train espanso da {len(train_data_list)} "
            f"a {len(train_dataset)} grafi (Originali + SO(3) Trasformati ad ogni epoca)"
        )
    else:
        train_dataset = train_data_list

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    net = define_target_neural_net(model_name, backend_name, dataset_name)
    est = NeuralEstimator(
        net,
        gcfg,
        backend_name,
        ranking_fn=lambda yt, yp: multitask_ranking_metrics(yt, yp, task_cols, k=50),
    )
    history = est.fit(train_loader, val_loader)
    preds = est.predict_proba(test_loader)
    est.save(configger.paths["models_output_dir"] / f"{backend_name}_{model_name}_seed{seed}")
    return preds, history.to_dict()

def main():
    args = parse_args()
    setup_logging(args.model, args.seed, args.backend)
    set_seed(args.seed)
    logger.info(f"=== RUN model={args.backend}_{args.model} seed={args.seed} k={args.k} ===")

    paths = configger.paths
    runs_dir = paths["models_output_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths["preprocessed_input_dir"] / "molecules.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"File parquet non trovato: {parquet_file}. Esegui prima lo Step 02.")

    train_df, val_df, test_df, task_cols = load_dataset_views(parquet_file)
    y_test = test_df[task_cols].to_numpy()

    if args.model == "xgboost":
        preds, history = run_xgboost(train_df, val_df, test_df, task_cols, args.seed)
    else:
        preds, history = run_neural(train_df, val_df, test_df, task_cols, args.seed, args.model, args.backend, 'nci')

    report = ranking_report(y_test, preds, task_cols, args.k)
    report.update({"model": args.model, "seed": args.seed, "k_values": args.k})
    if history is not None:
        report["history"] = history
        report["best_epoch"] = int(np.argmin(history["val_loss"])) + 1

    out = runs_dir / f"{args.backend}_{args.model}_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"ROC-AUC medio: {report['roc_auc_mean']:.4f}")
    for k in args.k:
        logger.info(
            f"k={k}: P@k={report[f'k{k}']['precision_at_k']:.4f} | "
            f"EF@k={report[f'k{k}']['enrichment_at_k']:.4f} | AUPRC={report[f'k{k}']['auprc']:.4f}"
        )
    logger.info(f"Risultati salvati in {out}")


if __name__ == "__main__":
    main()