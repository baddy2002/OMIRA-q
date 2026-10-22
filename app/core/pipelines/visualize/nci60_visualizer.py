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
    """Visualizzatore per l'Exploratory Data Analysis (EDA) sul dataset NCI-60 (Supporta Tidy/Melted e Preprocessed Wide)."""

    def __init__(self, output_dir: str | Path = "data/output"):
        super().__init__(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Impostazioni grafiche Seaborn
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_distributions(self, df: pd.DataFrame) -> None:
        cols = [c for c in df.columns if c.startswith("pGI50_")]
        if not cols:
            logger.info("Nessuna colonna pGI50_ trovata. Skip distribuzioni.")
            return

        long = df[cols].melt(var_name="Linea", value_name="pGI50").dropna()
        long["Linea"] = long["Linea"].str.replace("pGI50_", "", regex=False)

        # 1. coverage per linea
        cov = long.groupby("Linea").size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(x=cov.index, y=cov.values, color="skyblue", ax=ax)
        ax.set_xticklabels(cov.index, rotation=90, fontsize=8)
        ax.set_ylabel("Numero di composti testati")
        ax.set_title("Copertura per linea cellulare")
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_coverage_per_cell_line.png", dpi=300)
        plt.close(fig)

        # 2. distribuzione globale di pGI50, con soglia > 5 (regola di progetto)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(long["pGI50"], bins=50, kde=True, color="teal", ax=ax)
        ax.axvline(x=5.0, color="red", linestyle="--", label="Soglia pGI50 = 5")
        ax.set_xlabel("pGI50")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_gi50_global_distribution.png", dpi=300)
        plt.close(fig)

        # 3. percentuale di attivi per linea (attivo: pGI50 > 5)
        active = long.assign(attivo=lambda d: d["pGI50"] > 5.0).groupby("Linea")["attivo"].mean() * 100
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
        cols = [c for c in df.columns if c.startswith("pGI50_")]
        if len(cols) < 2:
            logger.info("Meno di due linee: skip matrice di correlazione.")
            return

        # campione delle prime 15 linee, per leggibilità
        sample = df[cols[:15]].copy()
        sample.columns = [c.replace("pGI50_", "", 1) for c in sample.columns]

        corr = sample.corr(min_periods=50)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlazione tra risposte biologiche (prime 15 linee)")
        fig.tight_layout()
        fig.savefig(self.output_dir / "eda_cell_lines_correlation.png", dpi=300)
        plt.close(fig)

    def plot_preprocessed_data(self, df: pd.DataFrame) -> None:
        """Grafici dedicati al dataset preprocessato (Step 02 - formato Wide/QC Ready)."""
        y_cols = [c for c in df.columns if c.startswith("y_")]
        has_qc_data = "stress_2d" in df.columns or "split" in df.columns or len(y_cols) > 0

        if not has_qc_data:
            return

        logger.info("Rilevato dataset preprocessato: avvio grafici post-filtering e QPU mapping...")

        # 1. Distribuzione dello Stress 2D di Kruskal (Vincolo Atomi Neutri)
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

        # 2. Bilanciamento degli Split (Murcko Scaffold)
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

        # 3. Bilanciamento Attivi/Inattivi per le Linee Target (y_*)
        if y_cols:
            # Se ci sono troppi task, prendiamo solo quelli con dati effettivi o limitiamo la visualizzazione
            df_y = df[y_cols].melt(var_name="Task", value_name="Label").dropna()
            df_y["Task"] = df_y["Task"].str.replace("y_", "")
            df_y["Stato"] = df_y["Label"].map({1.0: "Attivo", 0.0: "Inattivo"})

            # Se ci sono troppi task, ordiniamo per frequenza o prendiamo i principali per pulizia visiva
            top_tasks = df_y["Task"].value_counts().index[:15] # Mostra al massimo i primi 15 più popolati se sono tanti
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