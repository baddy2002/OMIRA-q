# core/pipelines/load/nci60_loader.py
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

from app.core.pipelines.load.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class NCI60LegacyLoader(BaseLoader):
    """Loader Legacy per il dataset NCI-60.
    Legge il dataset NCI-60 grezzo, recupera da PubChem i CID a partire dai SID,
    scarica SMILES e ConnectivitySMILES per ciascun CID, ribalta il dataset in formato lungo
    e calcola la colonna binaria Is_Active.
    """

    def __init__(
        self,
        input_dir: str = "data/input",
        output_dir: str = "data/output",
        raw_file_name: str = "DTP_NCI60_RAW.xlsx",
        full_dataset_name: str = "nci60_full_dataset_OLD.csv",
        batch_size: int = 1000,
        activity_threshold: float = 5.0,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.raw_file_path = os.path.join(input_dir, raw_file_name)
        self.full_dataset_path = os.path.join(input_dir, full_dataset_name)
        self.batch_size = batch_size
        self.activity_threshold = activity_threshold

    def _post_request(self, url, payload, max_retries=5):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        for attempt in range(max_retries):
            try:
                r = requests.post(url, data=payload, headers=headers, timeout=60)
            except requests.RequestException as e:
                logger.error(f"[PUBCHEM EXCEPTION] {attempt+1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
                continue

            if r.status_code == 200:
                return r.json(), 200, r.text

            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 30 * (attempt + 1)))
                logger.warning(f"[PUBCHEM] 429, attendo {wait}s (tentativo {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            if r.status_code in (500, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue

            logger.warning(f"[PUBCHEM] HTTP {r.status_code}: {r.text[:300]}")
            return None, r.status_code, r.text

        raise RuntimeError(f"PubChem: retry esauriti su {url}. Probabile blocco IP, riprova più tardi.")

    def _fetch_sid_to_cid(self, sids: List[int]) -> Dict[int, int]:
        """Step 1: Mappa SID -> CID tramite PubChem."""
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/cids/JSON"
        sid_to_cid = {}

        for i in range(0, len(sids), self.batch_size):
            chunk = sids[i : i + self.batch_size]
            payload = {"sid": ",".join(map(str, chunk))}
            
            data, status, _ = self._post_request(url, payload)
            if data and "InformationList" in data and "Information" in data["InformationList"]:
                for item in data["InformationList"]["Information"]:
                    sid = item.get("SID")
                    cids = item.get("CID", [])
                    if sid and cids:
                        sid_to_cid[sid] = cids[0]  # Prende il primo CID associato al SID
            
            time.sleep(0.2)

        return sid_to_cid

    def _fetch_cid_to_smiles(self, cids: List[int]) -> Dict[int, Dict[str, Optional[str]]]:
        """Step 2: Mappa CID -> SMILES e ConnectivitySMILES tramite PubChem."""
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/SMILES,ConnectivitySMILES/JSON"
        cid_to_info = {}

        for i in range(0, len(cids), self.batch_size):
            chunk = cids[i : i + self.batch_size]
            payload = {"cid": ",".join(map(str, chunk))}
            
            data, status, _ = self._post_request(url, payload)
            if data and "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                for prop in data["PropertyTable"]["Properties"]:
                    cid = prop.get("CID")
                    smiles = prop.get("SMILES")
                    conn_smiles = prop.get("ConnectivitySMILES")
                    if cid:
                        cid_to_info[cid] = {
                            "SMILES": smiles,
                            "ConnectivitySMILES": conn_smiles,
                        }
            
            time.sleep(0.2)

        return cid_to_info

    def load_raw(self) -> pd.DataFrame:
        """Carica l'Excel grezzo, arricchisce con PubChem ed esegue l'unpivot delle linee tumorali."""
        if os.path.exists(self.full_dataset_path):
            logger.info(f"Trovato dataset in cache: {self.full_dataset_path}. Caricamento...")
            return pd.read_csv(self.full_dataset_path, low_memory=False)

        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"File grezzo non trovato in: {self.raw_file_path}")

        logger.info(f"Lettura file Excel grezzo: {self.raw_file_path}...")
        df = pd.read_excel(self.raw_file_path, header=9, na_values=["na", "NA", "-", ""])

        # Pulizia dei SID
        df["PubChem SID"] = pd.to_numeric(df["PubChem SID"], errors="coerce")
        df_valid_sid = df.dropna(subset=["PubChem SID"]).copy()
        df_valid_sid["PubChem SID"] = df_valid_sid["PubChem SID"].astype(int)

        unique_sids = df_valid_sid["PubChem SID"].unique().tolist()
        logger.info(f"Trovati {len(unique_sids)} PubChem SID unici. Avvio STEP 1 (SID -> CID)...")

        # STEP 1: SID -> CID
        sid_to_cid = self._fetch_sid_to_cid(unique_sids)
        logger.info(f"Mappati {len(sid_to_cid)}/{len(unique_sids)} SID in CID.")

        # Estrai CID unici per lo STEP 2
        unique_cids = list(set(sid_to_cid.values()))
        logger.info(f"Trovati {len(unique_cids)} CID unici. Avvio STEP 2 (CID -> SMILES)...")

        # STEP 2: CID -> SMILES & ConnectivitySMILES
        cid_to_info = self._fetch_cid_to_smiles(unique_cids)
        logger.info(f"Ottenute informazioni SMILES per {len(cid_to_info)}/{len(unique_cids)} CID.")

        # Mappatura su DataFrame originale
        df["PubChem CID"] = df["PubChem SID"].map(sid_to_cid)
        df["SMILES"] = df["PubChem CID"].map(lambda x: cid_to_info.get(x, {}).get("SMILES"))
        #df["ConnectivitySMILES"] = df["PubChem CID"].map(lambda x: cid_to_info.get(x, {}).get("ConnectivitySMILES"))

        # Unpivot linee tumorali
        cell_line_cols = [
            c for c in df.columns 
            if any(c.startswith(prefix) for prefix in ["BR:", "CNS:", "CO:", "LE:", "ME:", "LC:", "OV:", "PR:", "RE:"])
        ]
        
        # pivot a formato largo: una riga per NSC, colonne per linea
        wide = df[["NSC #", "SMILES"] + cell_line_cols].copy()
        wide = wide.rename(columns={"NSC #": "NSC"})
        wide = wide.dropna(subset=["NSC", "SMILES"])
        for c in cell_line_cols:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

        agg = wide.groupby(["NSC", "SMILES"], as_index=False)[cell_line_cols].mean()
        agg = agg.rename(columns={c: f"pGI50_{c}" for c in cell_line_cols})
        agg["NSC"] = agg["NSC"].astype(int)

        os.makedirs(self.input_dir, exist_ok=True)
        agg.to_csv(self.full_dataset_path, index=False)
        logger.info(f"Dataset legacy salvato in {self.full_dataset_path}")
        return agg

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Controllo del DataFrame estratto."""
        if "SMILES" in df.columns:
            df = df.dropna(subset=["SMILES"]).copy()
            logger.info(f"Record con SMILES valido: {len(df)}")
        return df

    def run(self) -> pd.DataFrame:
        df = self.load_raw()
        return self.check(df)