# scripts/01_etl_and_visualize.py
import logging
import sys
from app.core.pipelines.load.nci60_legacy_loader import NCI60LegacyLoader
from app.core.pipelines.load.nci60_loader import NCI60Loader
from app.core.pipelines.visualize.nci60_visualizer import NCI60Visualizer
from app.core.configs.configger import PipelineConfigger

configger = PipelineConfigger(config_root="app/core/configs")
# Configurazione del logging su console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("ETL_and_Visualize")


def main():
    logger.info("=== AVVIO ETL E VISUALIZZAZIONE EDA PER NCI-60 ===")

    # 1. Caricamento ed arricchimento dati da PubChem via Loader
    if configger.global_cfg.get("use_legacy_api_loader", False):
        logger.info("Utilizzo del loader legacy NCI60LegacyLoader per il dataset NCI-60.")
        loader = NCI60LegacyLoader(
            input_dir="data/input",
            output_dir="data/output",
            raw_file_name="DTP_NCI60_RAW.xlsx"
        )
    else:
        logger.info("Utilizzo del loader NCI60Loader per il dataset NCI-60.")
        loader = NCI60Loader(
            input_dir="data/input",
            output_dir="data/output",
            gi50_name="GI50.csv",
            smiles_name="nsc_smiles.csv",
            cancelled_name="GI50_Cancelled.csv",
            full_dataset_name="nci60_full_dataset.csv",
        )
    df = loader.run()

    logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape[0]} righe x {df.shape[1]} colonne.")
    logger.info(f"Esempio SMILES recuperati:\n{df[['NSC', 'SMILES']].head()}")

    # 2. Generazione grafici ed analisi esplorativa tramite Visualizer
    visualizer = NCI60Visualizer(output_dir="data/output")
    visualizer.run(df)

    logger.info("=== ESECUZIONE COMPLETATA CON SUCCESSO. Verificare i grafici in 'data/output/' ===")


if __name__ == "__main__":
    main()