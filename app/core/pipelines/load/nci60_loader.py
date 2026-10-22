# app/core/pipelines/load/nci60_loader.py
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from app.core.pipelines.load.base_loader import BaseLoader
from app.core.configs.configger import PipelineConfigger

configger = PipelineConfigger(config_root="app/core/configs")
logger = logging.getLogger(__name__)

WIDE_COLUMNS = ("NSC", "SMILES")


class NCI60Loader(BaseLoader):
    """Loader DTP per NCI-60 (release ottobre 2026).

    Input (in input_dir):
      - GI50.csv              : GI50 per esperimento (una riga per EXPID, NSC, linea)
      - nsc_smiles.csv        : SMILES per NSC
      - GI50_Cancelled.csv    : esperimenti cancellati per QC, da escludere

    Output: formato largo, una riga per NSC.
      Colonne: NSC, SMILES, pGI50_<linea>, cens_hi_<linea>, n_exp_<linea>, std_<linea>
    """
    __PANEL_TO_PREFIX = {
        "BRE": "BR", "CNS": "CNS", "COL": "CO", "LEU": "LE", "MEL": "ME",
        "LNS": "LC", "OVA": "OV", "PRO": "PR", "REN": "RE",
    }

    def __init__(
        self,
        input_dir: str = "data/input",
        output_dir: str = "data/output",
        gi50_name: str = "GI50.csv",
        smiles_name: str = "nsc_smiles.csv",
        cancelled_name: str = "GI50_Cancelled.csv",
        full_dataset_name: str = "nci60_full_dataset.csv",
        activity_threshold: float = 5.0,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.gi50_path = os.path.join(input_dir, gi50_name)
        self.smiles_path = os.path.join(input_dir, smiles_name)
        self.cancelled_path = os.path.join(input_dir, cancelled_name)
        self.full_dataset_path = os.path.join(input_dir, full_dataset_name)
        self.activity_threshold = activity_threshold

    def load_raw(self) -> pd.DataFrame:
        for path in (self.gi50_path, self.smiles_path, self.cancelled_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"File mancante: {path}")

        # 1. GI50 per esperimento, solo molare
        gi = pd.read_csv(
            self.gi50_path,
            usecols=[
                "EXPID", "NSC", "CONCENTRATION_UNIT", "LOG_HI_CONCENTRATION",
                "PANEL_CODE", "CELL_NAME", "COUNT", "AVERAGE", "STDDEV",
            ],
        )
        n_raw = len(gi)
        gi = gi[gi["CONCENTRATION_UNIT"] == "M"].copy()
        logger.info(f"GI50 righe totali: {n_raw}, in molare: {len(gi)}")

        # esclusione  linee non incluse in __PANEL_TO_PREFIX
        if configger.global_cfg.get("exclude_panels"):
            n_before = len(gi)
            gi = gi[~gi["PANEL_CODE"].isin(configger.global_cfg.get("exclude_panels"))].copy()
            logger.info(f"Righe escluse per pannello {configger.global_cfg.get('exclude_panels')}: {n_before - len(gi)}")

        # mappatura PANEL_CODE -> PREFIX
        gi["PREFIX"] = gi["PANEL_CODE"].map(self.__PANEL_TO_PREFIX)
        n_no_prefix = gi["PREFIX"].isna().sum()
        if n_no_prefix:
            logger.warning(f"Righe senza prefisso mappato, scartate: {n_no_prefix}")
        gi = gi.dropna(subset=["PREFIX"]).copy()

        # esclusione esperimenti cancellati
        cancelled = pd.read_csv(self.cancelled_path, usecols=["EXPID", "NSC", "CELL_NAME"])
        cancelled_keys = set(zip(cancelled["EXPID"], cancelled["NSC"], cancelled["CELL_NAME"]))
        is_cancelled = np.array(
            [(e, n, c) in cancelled_keys for e, n, c in zip(gi["EXPID"], gi["NSC"], gi["CELL_NAME"])],
            dtype=bool,
        )
        gi = gi[~np.array(is_cancelled, dtype=bool)].copy()
        logger.info(f"Esperimenti cancellati esclusi: {sum(is_cancelled)}")

        # pGI50 e censura superiore
        gi["pGI50"] = -gi["AVERAGE"]
        gi["cens_hi"] = np.isclose(gi["AVERAGE"], gi["LOG_HI_CONCENTRATION"], atol=1e-4)        
        # formato linea: PREFIX:CELL_NAME
        gi["LINE"] = gi["PREFIX"] + ":" + gi["CELL_NAME"].astype(str)

        #    pGI50: media sugli esperimenti NON censurati; se tutti censurati, media dei bordi
        #    cens_hi: True solo se TUTTI gli esperimenti sono censurati
        gi["pGI50_unc"] = gi["pGI50"].where(~gi["cens_hi"])
        agg = gi.groupby(["NSC", "LINE"]).agg(
            pGI50_unc=("pGI50_unc", "mean"),
            pGI50_all=("pGI50", "mean"),
            cens_hi=("cens_hi", "all"),
            n_exp=("EXPID", "nunique"),
            std=("pGI50", "std"),
        ).reset_index()
        agg["pGI50"] = agg["pGI50_unc"].fillna(agg["pGI50_all"])
        agg = agg.drop(columns=["pGI50_unc", "pGI50_all"])

        # forma larga: una riga per NSC
        wide = agg.pivot(index="NSC", columns="LINE", values=["pGI50", "cens_hi", "n_exp", "std"])
        wide.columns = [f"{metric}_{cell}" for metric, cell in wide.columns]
        wide = wide.reset_index()

        # SMILES per NSC (join solo su NSC)
        smi = pd.read_csv(self.smiles_path, usecols=["NSC", "SMILES"]).drop_duplicates("NSC")
        wide = wide.merge(smi, on="NSC", how="left")
        wide = wide[["NSC", "SMILES"] + [c for c in wide.columns if c not in ("NSC", "SMILES")]]

        logger.info(f"NSC nel dataset largo: {len(wide)}, senza SMILES: {wide['SMILES'].isna().sum()}")
        return wide

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in WIDE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonne mancanti: {missing}")
        if df["NSC"].duplicated().any():
            raise ValueError("NSC duplicati nel dataset largo")

        n_before = len(df)
        df = df.dropna(subset=["SMILES"]).copy()
        logger.info(f"NSC senza SMILES scartati: {n_before - len(df)}")
        return df

    def run(self) -> pd.DataFrame:
        df = self.check(self.load_raw())
        os.makedirs(self.input_dir, exist_ok=True)
        df.to_csv(self.full_dataset_path, index=False)
        logger.info(f"Dataset salvato in {self.full_dataset_path}")
        return df