# core/pipelines/load/base_loader.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseLoader(ABC):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Legge i file grezzi da input_dir, ritorna un DataFrame non processato."""
        ...

    @abstractmethod
    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Controlli minimi di sanità (colonne attese, NSC duplicati, ecc.)."""
        ...

    def run(self) -> pd.DataFrame:
        df = self.load_raw()
        return self.check(df)