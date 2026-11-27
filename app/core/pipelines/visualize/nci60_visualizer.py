# app/core/pipelines/visualize/nci60_visualizer.py
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from app.core.pipelines.visualize.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class NCI60Visualizer(BaseVisualizer):
    """
    Visualizzatore per l'Exploratory Data Analysis (EDA) sul dataset NCI-60 
    Supporta Tidy/Melted e Preprocessed Wide).
    """

    def __init__(self, output_dir: str | Path = "data/output", threshold: float = 5.0):
        super().__init__(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Impostazioni grafiche Seaborn
        sns.set_theme(style="whitegrid", palette="muted")
        self.threshold = threshold

    def plot_distributions(self, df: pd.DataFrame) -> None:
        """
        stampa grafici per capire: 
         - quanti esempi ci sono per ogni linea
         - quanti di questi sono attivi o inattivi
         - i valori di GI50 presenti nel dataset per indagare su soglie di default o altri valori sospetti
        """
        # prendi valori GI50 per tutte le linee
        cols = [c for c in df.columns if c.startswith("pGI50_")]
        if not cols:
            logger.info("Nessuna colonna pGI50_ trovata. Skip distribuzioni.")
            return

        # metti le linee in righe
        long = df[cols].melt(var_name="Linea", value_name="pGI50").dropna()
        long["Linea"] = long["Linea"].str.replace("pGI50_", "", regex=False)

        # numero campioni testati su ogni linea
        cov = long.groupby("Linea").size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(x=cov.index, y=cov.values, color="skyblue", ax=ax)
        ax.set_xticklabels(cov.index, rotation=90, fontsize=8)
        ax.set_ylabel("Numero di composti testati")
        ax.set_title("Copertura per linea cellulare")
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_coverage_per_cell_line.png", dpi=300)
        plt.close(fig)

        # istogramma con i valori di gi50
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(long["pGI50"], bins=50, kde=True, color="teal", ax=ax)
        ax.axvline(x=self.threshold, color="red", linestyle="--", label="Soglia pGI50 = 5")
        ax.set_xlabel("pGI50")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_gi50_global_distribution.png", dpi=300)
        plt.close(fig)

        # percentuale composti attivi per ogni linea cellulare
        active = long.assign(attivo=lambda d: d["pGI50"] > self.threshold).groupby("Linea")["attivo"].mean() * 100
        active = active.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(x=active.index, y=active.values, color="salmon", ax=ax)
        ax.axhline(y=50, color="black", linestyle=":")
        ax.set_xticklabels(active.index, rotation=90, fontsize=8)
        ax.set_ylabel("% attivi (pGI50 > 5)")
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_active_molecules_ratio.png", dpi=300)
        plt.close(fig)

    def plot_correlations(self, df: pd.DataFrame) -> None:
        """
        stampa grafico di correlazione tra le prime 15 linee
        """
        cols = [c for c in df.columns if c.startswith("pGI50_")]
        if len(cols) < 2:
            logger.info("Meno di due linee: skip matrice di correlazione.")
            return

        # campione delle prime 15 linee, per leggibilità
        sample = df[cols[:15]].copy()
        sample.columns = [c.replace("pGI50_", "", 1) for c in sample.columns]
        # calcola correlazione se ci sono almeno 50 righe in comune in cui i campioni non sono NaN
        corr = sample.corr(min_periods=50)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlazione tra risposte biologiche (prime 15 linee)")
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_cell_lines_correlation.png", dpi=300)
        plt.close(fig)

    def plot_preprocessed_data(self, df: pd.DataFrame) -> None:
        """
        Grafici dedicati al dataset preprocessato
        """
        y_cols = [c for c in df.columns if c.startswith("y_")]
        has_qc_data = "stress_2d" in df.columns or "split" in df.columns or len(y_cols) > 0

        if not has_qc_data:
            return

        logger.info("Rilevato dataset preprocessato: avvio grafici post-filtering e QPU mapping...")

        # distribuzione dello Stress 2D di Kruskal (layout ad atomi neutri 2D)
        if "stress_2d" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df["stress_2d"], bins=30, kde=True, color="teal", ax=ax)
            ax.axvline(x=0.15, color="red", linestyle="--", linewidth=2, label="Soglia Max Stress (0.15)")
            ax.set_xlabel("Stress 2D (Kruskal Stress-1)")
            ax.set_ylabel("Numero Molecole")
            ax.set_title("Distribuzione dello Stress 2D di Embedding (Registro QPU)")
            ax.legend()
            fig.tight_layout()
            save_path = self.output_dir / "eda_stress_2d_distribution.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Grafico Stress 2D salvato in: {save_path}")

        # bilanciamento degli Split (Murcko Scaffold)
        if "split" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="split", hue="split", palette="viridis", legend=False, ax=ax)
            ax.set_xlabel("Split")
            ax.set_ylabel("Conteggio Molecole")
            ax.set_title("Distribuzione Molecole per Scaffold Split")
            fig.tight_layout()
            save_path = self.output_dir / "eda_scaffold_splits.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Grafico Scaffold Split salvato in: {save_path}")

        # bilanciamento Attivi/Inattivi per le Linee Target (y_*)
        if y_cols:
            # se ci sono troppi task, prendiamo solo quelli con dati effettivi o limitiamo la visualizzazione
            df_y = df[y_cols].melt(var_name="Task", value_name="Label").dropna()
            df_y["Task"] = df_y["Task"].str.replace("y_", "")
            df_y["Stato"] = df_y["Label"].map({1.0: "Attivo", 0.0: "Inattivo"})

            # se ci sono troppi task, mostra al massimo i primi 15 più popolati se sono tanti
            top_tasks = df_y["Task"].value_counts().index[:15] 
            df_y_filtered = df_y[df_y["Task"].isin(top_tasks)]

            fig, ax = plt.subplots(figsize=(14, 6))
            sns.countplot(data=df_y_filtered, x="Task", hue="Stato", palette="Set1", ax=ax)
            ax.set_xlabel("Linea Cellulare Target")
            ax.set_ylabel("Numero Molecole")
            ax.set_title("Bilanciamento Attivi / Inattivi nei Task Principali Selezionati")
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            save_path = self.output_dir / "eda_target_activity_distribution.jpg"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Grafico bilanciamento attivi salvato in: {save_path}")

    def run(self, df: pd.DataFrame) -> None:
        """Esegue l'intera pipeline di visualizzazione."""
        self.plot_distributions(df)
        self.plot_correlations(df)
        self.plot_preprocessed_data(df)