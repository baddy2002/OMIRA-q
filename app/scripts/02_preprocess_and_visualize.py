# app/scripts/02_preprocess_and_visualize.py
import logging
import sys
from pathlib import Path
import pandas as pd

from app.core.configs.configger import PipelineConfigger
from app.core.pipelines.preprocess.nci60_preprocessor import NCI60PreProcessor
from app.core.pipelines.visualize.nci60_visualizer import NCI60Visualizer

# Configurazione del logging (Console + File di Debug in data/debug/preprocess.log)
debug_dir = Path("data/debug")
debug_dir.mkdir(parents=True, exist_ok=True)
log_file_path = debug_dir / "preprocess.log"

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler, file_handler],
)

logger = logging.getLogger("Preprocess_and_Visualize")


def main():
    logger.info("=== AVVIO PREPROCESSING E VISUALIZZAZIONE POST-FILTERING PER NCI-60 ===")
    logger.info(f"Log dettagliati salvati in: {log_file_path}")

    # 1. Inizializzazione del configuratore centrale
    configger = PipelineConfigger(config_root="app/core/configs")
    paths = configger.paths
    nci_cfg = configger.get_data_config("nci")

    # Assicura che le cartelle definite nel config.yml esistano
    paths["preprocessed_input_dir"].mkdir(parents=True, exist_ok=True)
    paths["preprocessed_output_dir"].mkdir(parents=True, exist_ok=True)

    input_file = paths["input_dir"] / "nci60_full_dataset.csv"

    if not input_file.exists():
        logger.error(
            f"File non trovato: {input_file}. "
            f"Eseguire prima lo script app/scripts/01_etl_and_visualize.py"
        )
        sys.exit(1)

    # 2. Caricamento dataset intermedio dallo Step 1
    logger.info(f"Caricamento dataset intermedio da: {input_file}")
    df_raw = pd.read_csv(input_file, low_memory=False)

    # 3. Esecuzione del Preprocessing (Funnel, RDKit, Conformeri 3D, Stress 2D, Split Scaffold)
    logger.info("Inizializzazione Preprocessor con parametri chimico-quantistici da nci.yaml...")
    preprocessor = NCI60PreProcessor(
        input_dir=str(paths["input_dir"]),
        output_dir=str(paths["preprocessed_input_dir"]),
        cfg=nci_cfg,
    )

    logger.info("Esecuzione pipeline di Preprocessing...")
    df_prep = preprocessor.run(df_raw)

    logger.info(f"Dataset preprocessato salvato in: '{paths['preprocessed_input_dir']}'")
    logger.info(f"Molecole finali pronti per il training: {len(df_prep)} righe x {df_prep.shape[1]} colonne.")

    # 4. Generazione dei grafici ed analisi esplorativa post-preprocessing
    logger.info("Generazione dei grafici ed EDA post-preprocessing...")
    visualizer = NCI60Visualizer(output_dir=str(paths["preprocessed_output_dir"]))
    visualizer.run(df_prep)

    # Stampa di sintesi del Funnel di Scarto
    logger.info("=== RIASSUNTO FUNNEL DI SCARTO ===")
    for step in preprocessor.funnel:
        logger.info(
            f"Step [{step['step']}]: In={step['in']} -> Out={step['out']} "
            f"(Scartati: {step['in'] - step['out']} | Motivi: {step['reasons']})"
        )

    logger.info(
        f"=== ESECUZIONE COMPLETATA CON SUCCESSO. "
        f"Dataset pronti in '{paths['preprocessed_input_dir']}' e grafici in '{paths['preprocessed_output_dir']}' ==="
    )


if __name__ == "__main__":
    main()