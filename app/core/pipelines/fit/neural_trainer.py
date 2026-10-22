# app/core/pipelines/fit/neural_trainer.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from app.core.models.history import History

logger = logging.getLogger(__name__)


class NeuralTrainer:
    """Trainer universale per reti neurali a grafo (GATv2, EGAT) con supporto multi-task e NaN masking."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        epochs: int = 100,
        patience: int = 10,
        max_grad_norm: float | None = 1.0,
        ranking_fn=None,  # callable(y_true: np.ndarray, y_pred: np.ndarray) -> dict, solo diagnostica
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.ranking_fn = ranking_fn
        self.pos_weight: torch.Tensor | None = None

    @torch.no_grad()
    def _compute_pos_weight(self, loader: DataLoader) -> torch.Tensor:
        """pos_weight per task = n_negativi / n_positivi, contati sul train (NaN esclusi).

        Calcolato una sola volta, prima del training, e tenuto fisso.
        """
        n_pos, n_neg = None, None
        for batch in loader:
            y = batch.y
            mask = ~torch.isnan(y)
            pos = ((y == 1) & mask).sum(dim=0).float()
            neg = ((y == 0) & mask).sum(dim=0).float()
            n_pos = pos if n_pos is None else n_pos + pos
            n_neg = neg if n_neg is None else n_neg + neg

        # task senza positivi: peso 1.0 per evitare divisione per zero
        weight = torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1.0), torch.ones_like(n_pos))
        logger.info(f"pos_weight per task: {[round(w, 3) for w in weight.tolist()]}")
        return weight.to(self.device)
    
    def compute_masked_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        if not mask.any():
            return preds.sum() * 0.0

        # i NaN vanno sostituiti PRIMA della BCE: il backward di BCE moltiplica per y
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        raw_loss = F.binary_cross_entropy_with_logits(
            preds,
            safe_targets,
            pos_weight=self.pos_weight,  # shape [num_tasks], broadcast sulla dimensione batch
            reduction="none",
        )
        masked = raw_loss * mask.float()
        return masked.sum() / mask.float().sum().clamp(min=1.0)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Ritorna (val_loss_pesata, val_loss_non_pesata).

        La pesata decide il checkpoint; la non pesata serve al confronto con i run precedenti.
        """
        self.model.eval()
        total_w, total_u = 0.0, 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            total_w += self.compute_masked_loss(preds, batch.y).item()

            # stessa loss senza pesi, calcolata con pos_weight temporaneamente disattivato
            saved = self.pos_weight
            self.pos_weight = None
            total_u += self.compute_masked_loss(preds, batch.y).item()
            self.pos_weight = saved
            n_batches += 1
        n = max(n_batches, 1)
        return total_w / n, total_u / n

    @torch.no_grad()
    def collect_predictions(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Predizioni (probabilità) e target del loader, in ordine. Solo per diagnostica."""
        self.model.eval()
        ys, ps = [], []
        for batch in loader:
            batch = batch.to(self.device)
            probs = torch.sigmoid(self.model(batch))
            ps.append(probs.cpu().numpy())
            ys.append(batch.y.cpu().numpy())
        return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            preds = self.model(batch)
            targets = batch.y  # Assumendo batch.y con shape [batch_size, n_tasks]
            
            loss = self.compute_masked_loss(preds, targets)
            
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            self.optimizer.step()

            #if loss.requires_grad:
                #loss.backward()

                #evita esplosione del gradiente con il clipping
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                #self.optimizer.step()
                
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> History:
        history = History()
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        self.pos_weight = self._compute_pos_weight(train_loader)

        logger.info(f"Avvio training neurale per {self.epochs} epoche (Device: {self.device})...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss_w, val_loss_u = self.evaluate(val_loader)
            
            rank = {}
            if self.ranking_fn is not None:
                y_val, p_val = self.collect_predictions(val_loader)
                rank = self.ranking_fn(y_val, p_val)  # solo diagnostica

            history.update(
                train_loss,
                val_loss_w,
                val_precision_at_k=rank.get("mean_precision_at_k"),
                val_auprc=rank.get("mean_auprc"),
            )

            log_rank = (
                f" | Val P@k: {rank['mean_precision_at_k']:.4f} | Val AUPRC: {rank['mean_auprc']:.4f}"
                if rank and rank.get("mean_precision_at_k") is not None and rank.get("mean_auprc") is not None
                else ""
            )
            logger.info(
                f"Epoca {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} | "
                f"Val Loss (pesata): {val_loss_w:.4f} | Val Loss (non pesata): {val_loss_u:.4f}"
                + log_rank
            )

            # Checkpoint e early stopping sulla val loss PESATA
            if val_loss_w < best_val_loss:
                best_val_loss = val_loss_w
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping attivato all'epoca {epoch}.")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history