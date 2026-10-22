# core/pipelines/visualize/base_visualizer.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseVisualizer(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    @abstractmethod
    def plot_distributions(self, df: pd.DataFrame) -> None:
        """Distribuzioni delle feature chiave (es. GI50, num_atomi)."""
        ...

    @abstractmethod
    def plot_correlations(self, df: pd.DataFrame) -> None:
        """Relazioni tra feature strutturali e GI50."""
        ...

    def run(self, df: pd.DataFrame) -> None:
        self.plot_distributions(df)
        self.plot_correlations(df)