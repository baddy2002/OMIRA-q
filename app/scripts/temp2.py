# app/scripts/temp.py
from __future__ import annotations
import argparse
import importlib
import logging
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Importa dinamicamente lo script 03 esistente
script03 = importlib.import_module("app.scripts.03_classical_train_and_compare")

def main():
    args = script03.parse_args()
    script03.setup_logging(args.model, args.seed, args.backend)
    script03.set_seed(args.seed)
    
    logger = logging.getLogger("Classical_Train_and_Compare")
    logger.info(f"=== SANITY CHECK OVERFIT (64 campioni) - FASE 1 model={args.backend}_{args.model} seed={args.seed} ===")

    paths = script03.configger.paths
    parquet_file = paths["preprocessed_input_dir"] / "molecules.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"File parquet non trovato: {parquet_file}")

    # 1. Carica i dati usando la funzione originale dello script 03
    train_df, val_df, test_df, task_cols = script03.load_dataset_views(parquet_file)

    # 2. Taglio chirurgico: riduce tutto a 64 campioni
    logger.info(">>> ISOLAMENTO DI SOLI 64 CAMPIONI PER OVERFITTING TEST <<<")
    train_df = train_df.head(64).reset_index(drop=True)
    val_df = train_df.head(64).reset_index(drop=True)
    test_df = test_df.head(64).reset_index(drop=True)

    y_test = test_df[task_cols].to_numpy()

    if args.model == "xgboost":
        logger.info("Il sanity check è pensato per le reti neurali.")
        return

    # 3. Esegue il training supervisionato diretto
    preds, history = script03.run_neural(train_df, val_df, test_df, task_cols, args.seed, args.model, args.backend, 'nci')

    logger.info("Sanity Check Fase 1 completato.")

if __name__ == "__main__":
    main()