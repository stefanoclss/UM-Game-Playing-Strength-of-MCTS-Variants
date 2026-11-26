"""Exploratory Data Analysis & Visual Diagnostics (`pipeline/eda.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Target Variable: `utility_agent1` (regression | RMSE)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def _to_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Convert Polars or Pandas input frame to a clean Pandas DataFrame."""
    if isinstance(df, pl.DataFrame):
        return pd.DataFrame(df.to_dict(as_series=False))
    return df.copy()


def profile_dataset(
    df: pd.DataFrame | pl.DataFrame,
    target_col: str = "utility_agent1",
) -> dict[str, Any]:
    """Compute row/column cardinality, missingness rates, and target distribution moments."""
    pdf = _to_pandas(df)
    n_rows = max(len(pdf), 1)
    null_rates = {
        str(col): round(float(pdf[col].isna().sum()) / n_rows, 4) for col in pdf.columns
    }
    num_cols = [str(c) for c in pdf.select_dtypes(include=["number", "bool"]).columns]
    cat_cols = [str(c) for c in pdf.columns if str(c) not in num_cols]
    summary: dict[str, Any] = {
        "rows": int(len(pdf)),
        "columns": int(pdf.shape[1]),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "top_missing_columns": dict(
            sorted(null_rates.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
    }
    if target_col in pdf.columns:
        y_num = pd.to_numeric(pdf[target_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(y_num) > 0:
            summary["target_stats"] = {
                "mean": round(float(np.mean(y_num)), 6),
                "std": round(float(np.std(y_num)), 6),
                "min": round(float(np.min(y_num)), 6),
                "p50": round(float(np.median(y_num)), 6),
                "max": round(float(np.max(y_num)), 6),
            }
    return summary


def plot_eda_overview(
    df: pd.DataFrame | pl.DataFrame,
    target_col: str = 'utility_agent1',
    output_path: str | Path = "artifacts/eda_overview.png",
) -> Path:
    """Render a 4-panel Executive Data Hygiene & Feature Correlation Dashboard."""
    pdf = _to_pandas(df)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), facecolor="#F8FAFC")
    fig.suptitle(
        f"Exploratory Data Forensics — {len(pdf):,} Rows × {pdf.shape[1]} Columns",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.40, color="#CBD5E1")

    # 1. Top Missingness (%)
    miss = (pdf.isna().mean() * 100.0).sort_values(ascending=False).head(12)
    axes[0, 0].barh(
        [str(c)[:26] for c in miss.index[::-1]],
        miss.values[::-1],
        color="#0284C7",
        alpha=0.88,
    )
    axes[0, 0].set_title("1. Top Column Missingness (%)", fontweight="bold", fontsize=10.5)
    axes[0, 0].set_xlabel("% Null Rows")

    # 2. Target Distribution & Moments
    if target_col in pdf.columns:
        y_vals = pd.to_numeric(pdf[target_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        n_bins = min(35, max(6, len(y_vals) // 5))
        axes[0, 1].hist(y_vals, bins=n_bins, color="#059669", alpha=0.86, edgecolor="white")
        if len(y_vals) >= 2:
            axes[0, 1].axvline(
                float(np.median(y_vals)),
                color="#0F172A",
                linestyle="--",
                linewidth=1.4,
                label=f"Median={float(np.median(y_vals)):.3g}",
            )
            axes[0, 1].legend(fontsize=8.5, frameon=True)
        axes[0, 1].set_title(
            f"2. Target Distribution (`{target_col}`)", fontweight="bold", fontsize=10.5
        )
    else:
        axes[0, 1].set_title("2. Target Column Not Present", fontweight="bold", fontsize=10.5)

    # 3. Schema Modality Composition
    num_df = pdf.select_dtypes(include=["number", "bool"])
    n_num = int(num_df.shape[1])
    n_cat = int(pdf.shape[1] - n_num)
    n_complete = int((pdf.isna().sum() == 0).sum())
    axes[1, 0].barh(
        ["Complete Columns (0% Null)", "Categorical Columns", "Numeric / Boolean Columns"],
        [n_complete, n_cat, n_num],
        color=["#10B981", "#F59E0B", "#6366F1"],
        alpha=0.88,
    )
    axes[1, 0].set_title(
        "3. Schema Cardinality & Hygiene Breakdown", fontweight="bold", fontsize=10.5
    )
    axes[1, 0].set_xlabel("Column Count")

    # 4. Top Numeric Feature Correlation Matrix (compact F1..F6 ticks, zero label overlap)
    cand_num = [c for c in num_df.columns if str(c) != target_col][:6]
    axes[1, 1].grid(False)
    if len(cand_num) >= 2:
        corr_mat = num_df[cand_num].corr().fillna(0.0).to_numpy(dtype=np.float64)
        short_codes = [f"F{i + 1}" for i in range(len(cand_num))]
        im = axes[1, 1].imshow(corr_mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axes[1, 1].set_xticks(range(len(short_codes)), short_codes)
        ylabs = [f"F{i + 1}: {str(c)[:18]}" for i, c in enumerate(cand_num)]
        axes[1, 1].set_yticks(range(len(ylabs)), ylabs, fontsize=8.5)
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        axes[1, 1].set_title(
            "4. Numeric Predictor Correlation Matrix", fontweight="bold", fontsize=10.5
        )
    else:
        axes[1, 1].set_title("4. Insufficient Numeric Columns for Correlation", fontsize=10.5)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is not None and out.is_file():
            display(Image(filename=str(out)))
    except ImportError:
        pass
    return out
