"""Top-Level Exploratory Data Analysis Orchestrator (`eda_pipeline.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Imports `pipeline.eda` to profile dataset schema, missingness, and visual target diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from pipeline.eda import plot_eda_overview, profile_dataset


def run_eda_pipeline(
    data_path: str | Path,
    target_col: str = "utility_agent1",
    plot_path: str | Path | None = "artifacts/eda_overview.png",
) -> dict[str, Any]:
    """Load tabular dataset from `data_path`, save visual overview, and return EDA summary."""
    path = Path(data_path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    report = profile_dataset(df, target_col=target_col)
    if plot_path is not None:
        saved_fig = plot_eda_overview(df, target_col=target_col, output_path=plot_path)
        report["eda_plot"] = str(saved_fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="sovLoop Modular EDA Orchestrator")
    parser.add_argument("--input", required=True, help="Path to CSV or Parquet dataset.")
    parser.add_argument("--target", default="utility_agent1", help="Target column name.")
    args = parser.parse_args()
    report = run_eda_pipeline(args.input, target_col=args.target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
