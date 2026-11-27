# app/scripts/temp.py
from __future__ import annotations
import argparse
import importlib
import logging
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from app.core.utils import backend

# Importa dinamicamente lo script 03 esistente
script03 = importlib.import_module("app.scripts.03_classical_train_and_compare")

def parse_args():
    p = argparse.ArgumentParser(description="Sanity Check Overfitting su 64 campioni")
    p.add_argument("--model", required=True, choices=["xgboost", "gatv2", "egnn3", "dmpnn"])
    p.add_argument("--backend", required=True, choices=["classical", "neutral_atom_sim", "neutral_atom_hw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, nargs="+", default=[10, 30])
    return p.parse_args()

def run_neural_cotraining(train_df, val_df, test_df, task_cols, seed: int, model_name: str, backend_name: str, dataset_name: str):
    import torch.nn.functional as F
    from torch_geometric.utils import to_dense_batch

    gcfg = script03.configger.get_model_config(model_name)
    
    # 1. Caricamento dati
    train_data_list = script03.df_to_pyg_data_list(train_df, task_cols, model_name)
    val_data_list = script03.df_to_pyg_data_list(val_df, task_cols, model_name)
    test_data_list = script03.df_to_pyg_data_list(test_df, task_cols, model_name)

    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_data_list, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    # 2. Inizializzazione modello
    net = script03.define_target_neural_net(model_name, backend_name, dataset_name)
    device = backend.get_optimal_device()
    net.to(device)

    # SEPARAZIONE NETTA DEGLI OPTIMIZER 
    gat_params = [p for name, p in net.named_parameters() if 'surrogate_net' not in name]
    optimizer_gat = torch.optim.Adam(gat_params, lr=1e-3)
    optimizer_surr = torch.optim.Adam(net.surrogate_net.parameters(), lr=1e-3)

    script03.logger.info(">>> INIZIO TRAINING CON STRATEGIA A (Co-Training Isolate) <<<")

    for epoch in range(50):
        net.train()
        epoch_loss = 0.0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            
            x_in = batch_data.x
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            batch_vector = batch_data.batch
            y_true = batch_data.y

            # --- 1. FORWARD DELLA GAT CON I PESI ATTUALI ---
            optimizer_gat.zero_grad()

            x = net.node_encoder(x_in)
            for conv in net.convs:
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=net.dropout, training=net.training)

            x_dense, mask = to_dense_batch(x, batch_vector, max_num_nodes=net.max_nodes)
            qubit_positions = net.compute_position_embeddings(x, batch_vector)
            x_projected = net.compute_waves_from_param(x_dense=x_dense, mask=mask)

            batch_size, num_nodes, params_per_node = x_projected.shape
            x_flat = x_projected.reshape(batch_size, num_nodes * params_per_node)
            pos_flat = qubit_positions.reshape(batch_size, -1)

            # --- 2. OTTHIENIAMO IL GROUND TRUTH DAL PULSER REALE (sulla nuova distribuzione) ---
            with torch.no_grad():
                true_probs = net.apply_quantum_layer(x_projected.detach(), mask, qubit_positions.detach())

            # --- 3. INNER ADAPTATION DEL SURROGATO (3-5 step locali sul batch corrente) ---
            # Questo costringe il surrogato ad allinearsi perfettamente alla nuova zona in cui si trova la GAT
            for p in net.surrogate_net.parameters():
                p.requires_grad = True

            for _ in range(3):  # 3 passaggi di adattamento locale
                optimizer_surr.zero_grad()
                surr_preds = net.surrogate_net(x_flat.detach(), pos_flat.detach())
                surr_loss = F.mse_loss(surr_preds, true_probs)
                surr_loss.backward()
                optimizer_surr.step()

            # --- 4. AGGIORNAMO LA GAT USANDO IL SURROGATO LOCALMENTE ADATTATO ---
            for p in net.surrogate_net.parameters():
                p.requires_grad = False  # Congeliamo il surrogato per non alterare i pesi mentre la GAT impara

            qubit_probs_gat = net.surrogate_net(x_flat, pos_flat)
            logits = torch.logit(qubit_probs_gat.clamp(min=1e-6, max=1.0 - 1e-6))
            
            # Gestione task loss biologica con maschera NaN
            mask = ~torch.isnan(y_true)
            safe_targets = torch.where(mask, y_true, torch.zeros_like(y_true))
            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                safe_targets,
                reduction='none'
            )
            masked = raw_loss * mask.float()
            task_loss = masked.sum() / mask.float().sum().clamp(min=1.0)

            task_loss.backward()
            optimizer_gat.step()

            # --- BLOCCO DI DEBUG / DUMP SU CSV ---
            import pandas as pd
            
            with torch.no_grad():
                # Estraiamo i parametri usando gli attributi nativi della rete
                w_param = getattr(net, 'waves_param', 4)
                p_size = getattr(net, 'phase_param_size', 1)
                
                amp_coeffs = x_projected[..., :w_param].cpu().numpy()
                det_coeffs = x_projected[..., w_param : 2 * w_param].cpu().numpy()
                phase_raw = x_projected[..., 2 * w_param : 2 * w_param + p_size].cpu().numpy()
                
                pos_np = qubit_positions.cpu().numpy()
                probs_np = qubit_probs_gat.cpu().numpy()  # o qubit_probs a seconda della variabile attiva
                logits_np = logits.cpu().numpy()
                y_true_np = y_true.cpu().numpy()

                records = []
                batch_size = x_projected.shape[0]
                
                for i in range(batch_size):
                    records.append({
                        "sample_idx": i,
                        "amp_coeffs": amp_coeffs[i].tolist(),
                        "det_coeffs": det_coeffs[i].tolist(),
                        "phase_raw": phase_raw[i].tolist(),
                        "qubit_positions": pos_np[i].tolist(),
                        "qubit_probs": probs_np[i].tolist(),
                        "logits": logits_np[i].tolist(),
                        "y_true": y_true_np[i].tolist()
                    })
                    
                df_debug = pd.DataFrame(records)
                df_debug.to_csv("gat_pulser_debug_dump.csv", index=False)
                script03.logger.info(">>> DUMP DI DEBUG SALVATO CORRETTAMENTE IN gat_pulser_debug_dump.csv <<<")
                    
            epoch_loss += task_loss.item()

        script03.logger.info(f"Epoca {epoch+1}/50 | GAT Task Loss: {epoch_loss:.4f} | Surr MSE Loss: {surr_loss.item():.6f}")
    # Valutazione finale
    net.eval()
    preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = net(batch)
            preds_list.append(torch.sigmoid(out).cpu().numpy())
            
    preds = np.vstack(preds_list)
    return preds, {"train_loss": [], "val_loss": []}

def main():
    args = parse_args()
    script03.setup_logging(args.model, args.seed, args.backend)
    script03.set_seed(args.seed)
    
    logger = logging.getLogger("Classical_Train_and_Compare")
    logger.info(f"=== SANITY CHECK OVERFIT (64 campioni) model={args.backend}_{args.model} seed={args.seed} ===")

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

    # 3. Esegue il training con la funzione originale
    preds, history = run_neural_cotraining(
        train_df, val_df, test_df, task_cols, args.seed, args.model, args.backend, 'nci'
    )

    logger.info(f"Sanity Check completato. History Train Loss: {history.get('train_loss', [])}")

if __name__ == "__main__":
    main()