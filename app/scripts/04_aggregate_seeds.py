# app/scripts/04_aggregate_seeds.py
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from app.core.configs.configger import PipelineConfigger


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*_seed*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        row = {
            "model": r["model"],
            "seed": r["seed"],
            "roc_auc": r["roc_auc_mean"],
            "best_epoch": r.get("best_epoch"),
        }
        for k in r["k_values"]:
            row[f"p@{k}"] = r[f"k{k}"]["precision_at_k"]
            row[f"ef@{k}"] = r[f"k{k}"]["enrichment_at_k"]
            row["auprc"] = r[f"k{k}"]["auprc"]  # uguale per ogni k
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(values: pd.Series) -> tuple[float, float]:
    std = values.std(ddof=1) if len(values) > 1 else math.nan
    return values.mean(), std


def summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        rec = {"model": model, "n_seeds": len(g)}
        for m in metrics:
            if m in g:
                mu, sd = mean_std(g[m])
                rec[f"{m}_mean"] = round(mu, 4)
                rec[f"{m}_std"] = round(sd, 4) if not math.isnan(sd) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_diff(df: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    """Differenza a - b, accoppiata per seed (solo seed presenti in entrambi)."""
    ga = df[df.model == a].set_index("seed")[metric]
    gb = df[df.model == b].set_index("seed")[metric]
    common = ga.index.intersection(gb.index)
    d = ga[common] - gb[common]
    n = len(d)
    mu, sd = mean_std(d) if n else (math.nan, math.nan)
    t = mu / (sd / math.sqrt(n)) if n > 1 and sd > 0 else math.nan
    return {"metric": metric, "n": n, "diff_mean": round(mu, 4), "diff_std": round(sd, 4), "t": round(t, 2)}


def main():
    runs_dir = PipelineConfigger().paths["models_output_dir"] / "runs"
    df = load_runs(runs_dir)
    if df.empty:
        raise SystemExit(f"Nessun run trovato in {runs_dir}")

    metrics = ["roc_auc", "auprc", "p@50", "ef@50", "p@100", "ef@100"]
    metrics = [m for m in metrics if m in df.columns]

    print("\n=== Riepilogo per modello (media ± deviazione standard) ===")
    s = summary(df, metrics)
    print(s.to_string(index=False))

    print("\n=== Differenza GATv2 - XGBoost, accoppiata per seed ===")
    diffs = [paired_diff(df, "gatv2", "xgboost", m) for m in metrics]
    print(pd.DataFrame(diffs).to_string(index=False))

    print("\n=== Run singoli ===")
    print(df.to_string(index=False))

    out_csv = runs_dir / "summary.csv"
    s.to_csv(out_csv, index=False)
    print(f"\nRiepilogo salvato in {out_csv}")


if __name__ == "__main__":
    main()