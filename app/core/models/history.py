
from typing import Dict, List


class History:
    """Contenitore standardizzato per le metriche di training e validation."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_precision_at_k": [],
            "val_auprc": [],
        }

    def update(
        self,
        train_loss: float,
        val_loss: float,
        val_precision_at_k: float | None = None,
        val_auprc: float | None = None,
    ) -> None:
        self.metrics["train_loss"].append(float(train_loss))
        self.metrics["val_loss"].append(float(val_loss))
        if val_precision_at_k is not None:
            self.metrics["val_precision_at_k"].append(float(val_precision_at_k))
        if val_auprc is not None:
            self.metrics["val_auprc"].append(float(val_auprc))

    def to_dict(self) -> Dict[str, List[float]]:
        return self.metrics