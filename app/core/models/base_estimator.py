# app/core/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np

from app.core.models.history import History

class Estimator(ABC):
    """Interfaccia astratta universale per modelli classici (XGBoost) e neurali (GATv2/EGAT)."""

    @abstractmethod
    def fit(self, train_data: Any, val_data: Any) -> History:
        """Addestra il modello usando i dataset di train e validation."""
        pass

    @abstractmethod
    def predict_proba(self, data: Any) -> np.ndarray:
        """Ritorna le probabilità predette in formato numpy (shape: [n_samples, n_tasks])."""
        pass

    def predict(self, data: Any, threshold: float = 0.5) -> np.ndarray:
        """Ritorna le predizioni binarie basate su una soglia."""
        probs = self.predict_proba(data)
        return (probs >= threshold).astype(int)

    @abstractmethod
    def save(self, directory: Path | str) -> None:
        """Salva lo stato completo del modello in una directory dedicata."""
        pass

    @abstractmethod
    def load(self, directory: Path | str) -> None:
        """Carica lo stato del modello da una directory dedicata."""
        pass