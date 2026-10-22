# core/pipelines/preprocess/nci60_preprocessor.py
"""
Implementazione condivisa da tutti i dataset molecolari destinati al QC.
Restano astratti solo gli hook specifici del dataset:
standardize_table e build_labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

from app.core.pipelines.preprocess.base_preprocessor import BasePreprocessor


class NCI60PreProcessor(BasePreprocessor):
    transient_cols = ("mol", "xyz")

    def __init__(self, input_dir: str, output_dir: str, cfg: dict):
        super().__init__(input_dir, output_dir, cfg)
        # Estrazione diretta delle sezioni principali dello YAML
        # Se una sezione manca nello YAML, solleva immediatamente KeyError
        self.prep_cfg = self.cfg["preprocess"]
        self.chem_cfg = self.prep_cfg["chemistry"]
        self.qh_cfg = self.prep_cfg["quantum_hardware"]
        self.tasks_cfg = self.prep_cfg["tasks"]
        self.split_cfg = self.prep_cfg["split"]

    # ---------------- chimica ----------------
    def standardize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parsing, standardizzazione (frammento principale, neutralizzazione) e deduplicazione.

        Ogni trasformazione viene confrontata con la molecola di partenza, per registrare
        quante molecole cambiano forma. Le modifiche NON scartano righe: sono solo
        contate nel funnel, così si distingue tra perdita di dati e alterazione chimica.
        """
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        chooser = rdMolStandardize.LargestFragmentChooser()
        uncharger = rdMolStandardize.Uncharger()

        mols, reasons = [], []
        n_frag_changed = 0      # molecole in cui il frammento principale non è l'intera SMILES
        n_uncharged = 0         # molecole la cui forma cambia dopo la neutralizzazione
        smi_original, smi_after_frag, smi_after_uncharge = [], [], []

        for smi in df["smiles"]:
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if m is None:
                mols.append(None)
                reasons.append("smiles_invalid")
                smi_original.append(None)
                smi_after_frag.append(None)
                smi_after_uncharge.append(None)
                continue

            try:
                # 1. frammento principale: rimuove sali e contro-ioni
                m_frag = chooser.choose(m)
                # 2. neutralizzazione: carbossilato -> acido, ammonio -> ammina, ecc.
                m_unch = uncharger.uncharge(m_frag)
                # 3. verifica di validità chimica
                Chem.SanitizeMol(m_unch)

                s0 = Chem.MolToSmiles(m)
                s1 = Chem.MolToSmiles(m_frag)
                s2 = Chem.MolToSmiles(m_unch)
                if s1 != s0:
                    n_frag_changed += 1
                if s2 != s1:
                    n_uncharged += 1

                smi_original.append(s0)
                smi_after_frag.append(s1)
                smi_after_uncharge.append(s2)
                mols.append(m_unch)
                reasons.append(None)
            except Exception:
                mols.append(None)
                reasons.append("standardize_failed")
                smi_original.append(None)
                smi_after_frag.append(None)
                smi_after_uncharge.append(None)

        df = df.assign(mol=mols, smiles_std=smi_after_uncharge)
        df = self.gate(
            "standardize", df,
            pd.Series(reasons, index=df.index, dtype=object),
            stats={
                "frag_changed": n_frag_changed,
                "uncharged": n_uncharged,
            },
        )

        # --- deduplicazione per InChIKey27 completo ---
        # L'InChIKey completo restituisce solitamente 27 caratteri (es. AAABBB...-CCC...-N)
        df = df.assign(inchikey27=df["mol"].map(lambda m: Chem.MolToInchiKey(m)))

        pgi50_cols = [c for c in df.columns if c.startswith("pGI50_")]
        cens_cols = [c for c in df.columns if c.startswith("cens_hi_")]
        nexp_cols = [c for c in df.columns if c.startswith("n_exp_")]
        std_cols = [c for c in df.columns if c.startswith("std_")]

        dup_mask = df["inchikey27"].duplicated(keep="first")
        n_dup_groups = int(df.loc[dup_mask, "inchikey27"].nunique())

        n_conflict_groups = 0
        n_conflict_cells = 0
        merged_rows = {}

        for key, g in df.groupby("inchikey27", sort=False):
            if len(g) == 1:
                continue
            row = {}
            has_conflict_in_group = False
            
            # Media per i valori pGI50
            for c in pgi50_cols:
                vals = g[c].dropna()
                if vals.empty:
                    row[c] = np.nan
                else:
                    row[c] = vals.mean()  # <-- Media anziché il primo valore
                    if vals.nunique() > 1:
                        n_conflict_cells += 1
                        has_conflict_in_group = True

            # Per i flag di censura (AND logico: se è censurato in tutti/alcuni) o conteggi
            for c in cens_cols:
                vals = g[c].dropna()
                row[c] = vals.any() if not vals.empty else np.nan

            for c in nexp_cols:
                vals = g[c].dropna()
                row[c] = vals.sum() if not vals.empty else np.nan

            for c in std_cols:
                vals = g[c].dropna()
                row[c] = vals.mean() if not vals.empty else np.nan

            merged_rows[key] = row
            if has_conflict_in_group:
                n_conflict_groups += 1

        # Aggiorna la prima riga di ogni gruppo con i valori mediati
        for key, row in merged_rows.items():
            idx = df.index[(df["inchikey27"] == key)][0]
            for c, v in row.items():
                df.at[idx, c] = v

        return self.gate(
            "dedup", df,
            pd.Series(np.where(dup_mask, "duplicate", None), index=df.index, dtype=object),
            stats={
                "dup_groups": n_dup_groups,
                "dup_label_conflict_groups": int(n_conflict_groups),
                "dup_label_conflict_cells": int(n_conflict_cells),
            },
        )

    def filter_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        allowed = set(self.chem_cfg["allowed_elements"])
        n_min = self.chem_cfg["n_heavy_min"]
        n_max = self.chem_cfg["n_heavy_max"]
        max_deg = self.chem_cfg["max_degree"]

        def check(m):
            atoms = list(m.GetAtoms())
            if not {a.GetSymbol() for a in atoms} <= allowed:
                return "element_not_allowed"
            if not (n_min <= len(atoms) <= n_max):
                return "n_heavy_out_of_range"
            if max(a.GetDegree() for a in atoms) > max_deg:
                return "degree_too_high"
            return None

        df = df.assign(n_heavy=df["mol"].map(lambda m: m.GetNumAtoms()))
        return self.gate("chemistry", df, df["mol"].map(check).astype(object))

    # ---------------- task ----------------
    def select_tasks(self, df: pd.DataFrame) -> list[str]:
        """Sceglie i task (linee cellulari) privilegiando un buon bilanciamento
        di attivi/inattivi (es. fra 30% e 70%) e massimizzando l'indipendenza informativa (phi).
        """
        target_override = self.tasks_cfg.get("target_cell_lines")
        if target_override:
            return [t for t in target_override if f"y_{t}" in df.columns]

        min_samples = self.tasks_cfg["min_samples_per_task"]
        n_tasks_target = self.tasks_cfg["n_tasks"]

        # 1. Filtriamo i task che hanno almeno min_samples e un bilanciamento accettabile (es. 30% - 70% di attivi)
        valid_ok_tasks = []
        for t in self.tasks:
            y_col = f"y_{t}"
            if y_col not in df.columns:
                continue
            series = df[y_col].dropna()
            if len(series) < min_samples:
                continue
            
            # Calcola la percentuale di attivi (classe 1)
            active_ratio = series.mean()
            # Vincolo di bilanciamento desiderato (es. tra 0.30 e 0.70)
            if 0.30 <= active_ratio <= 0.70:
                valid_ok_tasks.append(t)

        # Se con i filtri ristretti non troviamo abbastanza task, allarghiamo un po' la forbice (es. 20% - 80%)
        if len(valid_ok_tasks) < (n_tasks_target or 1):
            for t in self.tasks:
                y_col = f"y_{t}"
                if y_col not in df.columns or t in valid_ok_tasks:
                    continue
                series = df[y_col].dropna()
                if len(series) >= min_samples and 0.20 <= series.mean() <= 0.80:
                    valid_ok_tasks.append(t)

        # Se ancora vuoto, prendiamo quelli con più campioni disponibili
        if not valid_ok_tasks:
            valid_ok_tasks = [t for t in self.tasks if df[f"y_{t}"].notna().sum() >= min_samples]

        if n_tasks_target is None or len(valid_ok_tasks) <= n_tasks_target:
            return valid_ok_tasks

        # 2. Selezione greedy basata sulla correlazione (indipendenza informativa) tra i task bilanciati
        y_cols = [f"y_{t}" for t in valid_ok_tasks]
        phi = df[y_cols].corr().fillna(0.0)
        phi.index = phi.columns = valid_ok_tasks

        # Partiamo dal task che ha il bilanciamento più vicino al 50% (active_ratio più vicino a 0.5)
        best_start = min(valid_ok_tasks, key=lambda t: abs(df[f"y_{t}"].mean() - 0.5))
        chosen = [best_start]

        while len(chosen) < n_tasks_target:
            rest = [t for t in valid_ok_tasks if t not in chosen]
            # Sceglie il task che minimizza la massima correlazione con quelli già scelti (massimizza la diversità)
            chosen.append(max(rest, key=lambda t: min(1 - abs(phi.loc[t, s]) for s in chosen)))

        return chosen

    # ---------------- geometria ----------------
    def _conformer_xyz(self, m):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        seed = self.split_cfg["seed"]
        mh = Chem.AddHs(m)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        if AllChem.EmbedMolecule(mh, p) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mh)
        except Exception:
            pass                                    # Se fallisce MMFF tiene le coordinate ETKDG
        mh = Chem.RemoveHs(mh)                      # l'ordine degli atomi pesanti non cambia
        return mh.GetConformer().GetPositions()

    @staticmethod
    def stress_2d(xyz: np.ndarray) -> float:
        """Stress-1 di Kruskal calcolato tramite ottimizzazione L-BFGS e gradiente analitico.
        Sostituisce sklearn.MDS evitando problemi di deprecazione e metriche errate."""
        n = len(xyz)
        if n <= 3:
            return 0.0  # Fino a 3 punti l'embedding 2D è isometricamente esatto
        
        d = squareform(pdist(xyz))
        den = (d ** 2).sum() / 2
        if den == 0:
            return 0.0
            
        c = xyz - xyz.mean(0)
        y0 = c @ np.linalg.svd(c, full_matrices=False)[2][:2].T

        def fg(v):
            y = v.reshape(n, 2)
            diff = y[:, None, :] - y[None, :, :]
            r = np.sqrt((diff ** 2).sum(-1))
            np.fill_diagonal(r, 1.0)
            e = r - d
            np.fill_diagonal(e, 0.0)
            return (e ** 2).sum() / 2, (2 * (e / r)[:, :, None] * diff).sum(1).ravel()

        res = minimize(fg, y0.ravel(), jac=True, method="L-BFGS-B")
        return float(np.sqrt(res.fun / den))

    def _stress_2d(self, xyz: np.ndarray) -> float:
        return self.stress_2d(xyz)
    
    def compute_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        max_stress = self.qh_cfg["max_stress_2d"]
        xyz = df["mol"].map(self._conformer_xyz)
        df = df.assign(xyz=xyz)
        df = self.gate(
            "conformer", 
            df, 
            pd.Series(np.where(xyz.isna(), "conformer_failed", None), index=df.index, dtype=object)
        )
        df = df.assign(stress_2d=df["xyz"].map(self._stress_2d))
        too = df["stress_2d"] > max_stress
        return self.gate(
            "stress_2d", 
            df, 
            pd.Series(np.where(too, "stress_2d_too_high", None), index=df.index, dtype=object)
        )
    
    # ---------------- campionamento / split ----------------
    def subsample(self, df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
        """Riduce il dataset a max_molecules, con campionamento stratificato.

        Gli strati sono la combinazione di numero di atomi pesanti (n_heavy) e
        presenza di almeno un'attività tra i task selezionati. Il campionamento
        mantiene la proporzione di attivi e la distribuzione di dimensione.

        Le molecole non campionate sono registrate nel funnel con motivo
        'subsampled_out', separato dagli scarti di qualità chimica.
        """
        max_mols = self.split_cfg["max_molecules"]
        seed = self.split_cfg["seed"]

        if max_mols is None or len(df) <= max_mols:
            # nessun campionamento necessario: funnel registrato comunque
            return self.gate(
                "subsample", df,
                pd.Series(None, index=df.index, dtype=object),
                stats={"requested_max": max_mols, "applied": False},
            )

        # attività: 1 se la molecola è attiva in almeno un task selezionato
        any_active = df[[f"y_{t}" for t in tasks]].fillna(0).max(axis=1).astype(int)
        strata = df["n_heavy"].astype(str) + "_" + any_active.astype(str)

        # campione: indici originali delle righe tenute
        kept_idx = (
            df.assign(_strata=strata)
              .groupby("_strata", group_keys=False)
              .sample(frac=max_mols / len(df), random_state=seed)
              .index
        )

        # motivo di scarto per ogni riga: None se tenuta, 'subsampled_out' se tolta
        reasons = pd.Series(
            np.where(df.index.isin(kept_idx), None, "subsampled_out"),
            index=df.index,
            dtype=object,
        )
        return self.gate(
            "subsample", df, reasons,
            stats={
                "requested_max": max_mols,
                "applied": True,
                "strata_count": int(strata.nunique()),
            },
        )

    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaf = df["mol"].map(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        groups = sorted(df.groupby(scaf).indices.values(), key=len, reverse=True)
        f_tr, f_va, _ = self.split_cfg["fractions"]
        n, n_tr, n_va = len(df), 0, 0
        split_arr = np.empty(n, dtype=object)
        
        for g in groups:
            if n_tr + len(g) <= f_tr * n:
                split_arr[g] = "train"
                n_tr += len(g)
            elif n_va + len(g) <= f_va * n:
                split_arr[g] = "val"
                n_va += len(g)
            else:
                split_arr[g] = "test"
                
        return df.assign(scaffold=scaf, split=split_arr)

    def prepare_for_save(self, df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
        # Mantieni solo le colonne di base + le y dei soli task selezionati dall'algoritmo greedy
        if tasks is not None:
            valid_task_cols = [f"y_{t}" for t in tasks]
            base_cols = [c for c in df.columns if not c.startswith("y_") or c in valid_task_cols]
            out = df[base_cols].copy()
        
        if "mol" in out.columns:
            out = out.drop(columns=["mol"])
        if "xyz" in out.columns and out["xyz"].dtype == object:
            out["xyz"] = out["xyz"].map(lambda a: a.tolist() if isinstance(a, np.ndarray) else a)
        return out

    # ---------------- stadi specifici per NCI-60 ----------------
    def standardize_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formato largo dal loader: una riga per NSC, colonne pGI50_<task>, cens_hi_<task>,
        n_exp_<task>, std_<task>. Il loader ha già mediato gli esperimenti: qui non si
        pivota e non si calcola nulla sui valori. Il nome del task è la parte dopo 'pGI50_'
        (es. 'CNS:SNB-75'), che diventa y_<task> in build_labels.
        """
        pcols = [c for c in df.columns if c.startswith("pGI50_")]
        if not pcols or not {"NSC", "SMILES"} <= set(df.columns):
            raise ValueError("Formato non riconosciuto: attese colonne NSC, SMILES, pGI50_<task>.")

        self.tasks = [c[len("pGI50_"):] for c in pcols]
        out = df.rename(columns={"SMILES": "smiles"}).copy()
        out["mol_id"] = "NSC_" + out["NSC"].astype(str)
        self.record("standardize_table", len(df), len(df), stats={"n_tasks": len(self.tasks)})
        return out


    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Etichette binarie y_<task> da pGI50_<task> e cens_hi_<task>.

        Per ogni (NSC, linea):
        - valore non censurato fuori da value_range -> NaN (errore di codifica)
        - censurato a concentrazione massima (cens_hi = True):
            pGI50 <= soglia -> 0 (inattivo a tutte le dosi testate)
            pGI50 >  soglia -> NaN (la dose massima era sotto soglia: non determinabile)
        - altrimenti etichetta = regola unica is_active (pGI50 > soglia)
        I valori pGI50 non vengono mai modificati: si derivano solo le etichette.
        """
        lo, hi = self.chem_cfg["value_range"]
        thr = self.chem_cfg["threshold"]
        out = df.copy()
        n_cells = n_out = n_undet = 0

        for task in self.tasks:
            v = pd.to_numeric(out[f"pGI50_{task}"], errors="coerce")
            cens = out[f"cens_hi_{task}"].astype(str).str.lower().eq("true")

            out_range = v.notna() & ~cens & ~v.between(lo, hi)
            undet = v.notna() & cens & (v > thr)

            n_cells += int(v.notna().sum())
            n_out += int(out_range.sum())
            n_undet += int(undet.sum())

            label = self.is_active(v, thr).astype(float).where(v.notna())
            out[f"y_{task}"] = label.mask(out_range | undet)

        self.record(
            "value_range", n_cells, n_cells - n_out - n_undet,
            reasons={"out_of_range_uncensored": n_out, "censored_undetermined": n_undet},
            stats={"range": [lo, hi], "threshold": thr},
        )
        return out

    @staticmethod
    def _smiles_or_none(self, m) -> str | None:
        """SMILES canonico di una molecola, oppure None se la molecola è None."""
        from rdkit import Chem
        return Chem.MolToSmiles(m) if m is not None else None

    def _identity_key(self, smi) -> str | None:
        """Chiave di identità: InChIKey27 dopo frammento principale e neutralizzazione.

        È la stessa identità usata dal dedup in standardize_molecules. Serve a decidere
        se due SMILES diversi sono lo stesso composto (stereoisomeri, forme con carica).
        Ritorna None se la SMILES non è standardizzabile.
        """
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        if not isinstance(smi, str):
            return None
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        try:
            m = rdMolStandardize.Uncharger().uncharge(
                rdMolStandardize.LargestFragmentChooser().choose(m)
            )
            Chem.SanitizeMol(m)
            return Chem.MolToInchiKey(m)
        except Exception:
            return None

    @staticmethod
    def is_active(gi50, thr: float):
        """Regola unica di attività: pGI50 strettamente maggiore della soglia.

        thr esatto è inattivo (bordo della serie). Arrotondamento a 6 decimali
        per evitare che il rumore di rappresentazione float cambi la classe.
        """
        return np.round(gi50, 6) > thr