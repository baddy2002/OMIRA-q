# app/core/pipelines/preprocess/base_preprocessor.py
"""
Contratto comune a tutti i dataset. Nessuna dipendenza da RDKit:
qui ci sono solo gli stadi (astratti), l'orchestrazione run() e
l'infrastruttura gate/funnel/salvataggio.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REJECT = "reject_reason"


class BasePreprocessor(ABC):
    # colonne non serializzabili (oggetti Mol, array) da togliere dagli scarti
    transient_cols: tuple = ()

    def __init__(self, input_dir: str, output_dir: str, cfg: Any):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.tasks: list[str] = []          # popolato da standardize_table
        self.funnel: list[dict] = []
        self.rejected: list[pd.DataFrame] = []

    # ---------------- stadi: specifici del dataset ----------------
    @abstractmethod
    def standardize_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Una riga per molecola (repliche aggregate). Colonne minime: mol_id, smiles
        + un valore grezzo per task. Deve popolare self.tasks."""
        ...

    @abstractmethod
    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge y_<task> in {0, 1, NaN} per ogni task in self.tasks."""
        ...

    # ---------------- stadi: dominio molecolare / QC ----------------
    @abstractmethod
    def standardize_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parsing, standardizzazione, dedup."""
        ...

    @abstractmethod
    def filter_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtri economici di fattibilita' (elementi, dimensione, grado...)."""
        ...

    @abstractmethod
    def select_tasks(self, df: pd.DataFrame) -> list[str]:
        """Sceglie i task (cell line) da tenere."""
        ...

    @abstractmethod
    def compute_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conformero 3D + validazione dell'embedding 2D (stadio costoso)."""
        ...

    @abstractmethod
    def subsample(self, df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
        ...

    @abstractmethod
    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge la colonna 'split' in {train, val, test}."""
        ...

    # ---------------- infrastruttura concreta ----------------
    def gate(
        self,
        name: str,
        df: pd.DataFrame,
        reasons: pd.Series,
        stats: dict | None = None,
    ) -> pd.DataFrame:
        """Filtra le righe in base a un motivo di scarto e registra il passaggio nel funnel.

        Parametri
        ----------
        name : str
            Nome dello stadio, usato nel funnel.
        df : DataFrame
            Tabella in ingresso allo stadio.
        reasons : Series
            Allineata a df. None = riga valida, stringa = motivo di scarto.
        stats : dict, opzionale
            Campi aggiuntivi per il funnel (es. conteggi di modifiche), già calcolati
            dallo stadio. Non sostituiscono 'in' e 'out', che restano sempre presenti.

        Ritorna
        -------
        DataFrame con le sole righe valide, indice ricostruito da zero.
        """
        bad = reasons.notna()
        entry = {
            "step": name,
            "in": len(df),                      # righe ricevute dallo stadio
            "out": int((~bad).sum()),           # righe che proseguono
            "reasons": reasons[bad].value_counts().to_dict(),  # scarti per motivo
        }
        if stats:
            entry.update(stats)
        self.funnel.append(entry)

        if bad.any():
            self.rejected.append(
                df.loc[bad].drop(columns=list(self.transient_cols), errors="ignore")
                  .assign(**{REJECT: reasons[bad].astype(str)})
            )
        return df.loc[~bad].reset_index(drop=True)

    def drop_unlabeled(self, df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
        has = df[[f"y_{t}" for t in tasks]].notna().any(axis=1)
        reasons = pd.Series(np.where(has, None, "no_label_in_tasks"), index=df.index, dtype=object)
        return self.gate("no_label", df, reasons)

    def prepare_for_save(self, df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
        return df

    def _make_serializable(self, obj: Any) -> Any:
        """Converte ricorsivamente Path e tipi non-JSON in tipi nativi."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        return obj

    def record(
        self,
        name: str,
        n_in: int,
        n_out: int,
        reasons: dict | None = None,
        stats: dict | None = None,
    ) -> None:
        """Registra nel funnel uno stadio che modifica valori ma non scarta righe.

        Usato per l'esclusione di misure (es. flip di etichetta): 'in' e 'out' contano
        misure, non molecole, e le misure escluse vengono salvate a parte.
        """
        entry = {"step": name, "in": int(n_in), "out": int(n_out), "reasons": reasons or {}}
        if stats:
            entry.update(stats)
        self.funnel.append(entry)

    def save(self, df: pd.DataFrame, tasks: list[str]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_for_save(df, tasks).to_parquet(self.output_dir / "molecules.parquet", index=False)
        
        if self.rejected:
            pd.concat(self.rejected, ignore_index=True).to_parquet(
                self.output_dir / "rejected.parquet", index=False
            )
            
        # Gestione flessibile sia per dict sia per oggetti di configurazione
        if isinstance(self.cfg, dict):
            raw_cfg = self.cfg
        elif hasattr(self.cfg, "data_configs"):
            raw_cfg = self.cfg.data_configs
        else:
            raw_cfg = getattr(self.cfg, "__dict__", str(self.cfg))

        cfg_serializable = self._make_serializable(raw_cfg)
        
        manifest = {
            "config": cfg_serializable, 
            "tasks": tasks, 
            "n_final": len(df),
            "config_hash": hashlib.sha256(json.dumps(cfg_serializable, sort_keys=True).encode()).hexdigest()[:12],
        }
        (self.output_dir / "funnel.json").write_text(json.dumps(self.funnel, indent=2))
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = self.standardize_table(df_raw)
        df = self.build_labels(df)
        df = self.standardize_molecules(df)
        df = self.filter_molecules(df)
        tasks = self.select_tasks(df)
        
        selected_y_cols = [f"y_{t}" for t in tasks]
        all_y_cols = [c for c in df.columns if c.startswith("y_")]
        cols_to_drop = [c for c in all_y_cols if c not in selected_y_cols]
        df = df.drop(columns=cols_to_drop)

        df = self.drop_unlabeled(df, tasks)
        df = self.compute_geometry(df)          # costoso: dopo i filtri economici
        df = self.subsample(df, tasks)
        df = self.split(df)
        self.save(df, tasks)
        return df