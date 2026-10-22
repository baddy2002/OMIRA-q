# app/core/models/xgb_estimator.py
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.core.models.base_estimator import Estimator
from app.core.models.history import History

class XGBEstimator(Estimator):
    """Estimator tabulare basato su XGBoost per un confronto baseline non geometrico."""

    is_neural: bool = False

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_params = config.get("params", {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42
        })
        # Usiamo XGBClassifier (gestisce nativamente i NaN se passati opportunamente)
        self.models: list[XGBClassifier | None] = []
        self.history = History()

    def fit(self, train_data: tuple[np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray]) -> History:
        X_train, y_train = train_data
        X_val, y_val = val_data

        n_tasks = y_train.shape[1] if y_train.ndim > 1 else 1
        if y_train.ndim == 1:
            y_train = y_train[:, None]
            if y_val is not None and y_val.ndim == 1:
                y_val = y_val[:, None]

        self.models = []
        all_train_losses = []
        all_val_losses = []

        for i in range(n_tasks):
            tr_valid = ~np.isnan(y_train[:, i])
            va_valid = ~np.isnan(y_val[:, i]) if val_data is not None else np.zeros(0, dtype=bool)

            if tr_valid.sum() < 10:
                self.models.append(None)
                continue

            model = XGBClassifier(**self.model_params)
            eval_set = [(X_val[va_valid], y_val[va_valid, i])] if va_valid.sum() > 0 else None

            model.fit(
                X_train[tr_valid], y_train[tr_valid, i],
                eval_set=eval_set,
                verbose=False
            )
            self.models.append(model)

            # Estrazione log history se disponibile dalle metriche di eval del modello
            results = model.evals_result()
            if results and "validation_0" in results:
                metric_name = list(results["validation_0"].keys())[0]
                t_metrics = results["validation_0"][metric_name]
                all_train_losses.append(t_metrics)
                if "validation_1" in results:
                    v_metrics = results["validation_1"][metric_name]
                    all_val_losses.append(v_metrics)

        # Se abbiamo tracciato le history, le aggreghiamo (media sui task) e le carichiamo nell'oggetto History
        if all_train_losses:
            max_len = max(len(l) for l in all_train_losses)
            # Padding o troncamento per allineare le epoche dei vari task
            padded_train = [np.pad(l, (0, max_len - len(l)), 'edge') for l in all_train_losses]
            mean_train_history = np.mean(padded_train, axis=0)

            mean_val_history = np.zeros_like(mean_train_history)
            if all_val_losses:
                padded_val = [np.pad(l, (0, max_len - len(l)), 'edge') for l in all_val_losses]
                mean_val_history = np.mean(padded_val, axis=0)

            for t_loss, v_loss in zip(mean_train_history, mean_val_history):
                self.history.update(float(t_loss), float(v_loss))
        else:
            self.history.update(0.0, 0.0)

        return self.history

    def predict_proba(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        preds_list = []
        for model in self.models:
            if model is None:
                preds_list.append(np.full((len(data),), np.nan))
                continue
            p = model.predict_proba(data)
            if p.ndim == 2 and p.shape[1] >= 2:
                preds_list.append(p[:, 1])
            else:
                preds_list.append(p)
        return np.stack(preds_list, axis=-1)

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, directory / "xgboost_models.joblib")

    def load(self, directory: Path | str) -> None:
        directory = Path(directory)
        model_path = directory / "xgboost_models.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modello XGBoost non trovato in: {model_path}")
        self.models = joblib.load(model_path)