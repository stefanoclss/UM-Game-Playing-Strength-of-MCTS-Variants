"""Top-Level Batch Inference Orchestrator (`inference_pipeline.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Loads zero-pickle `artifacts/model_artifact.json` (and `fold_*.lgb` boosters)
and executes `preprocessing(df)` -> `feature_engineering(df)` -> fold ensemble scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from pipeline.features import feature_engineering
from pipeline.models import predict_model_stage
from pipeline.postprocess import run_postprocess_stage
from pipeline.preprocess import preprocessing
from training_pipeline import (
    CONFIGURED_EVENT_COL,
    DEFAULT_ARTIFACT_PATH,
    _to_pandas,
    run_training_pipeline,
    save_trained_artifact,
)


def _load_frame(path_str: str | Path) -> pl.DataFrame:
    p = Path(path_str)
    return pl.read_parquet(p) if p.suffix.lower() == ".parquet" else pl.read_csv(p)


def load_trained_artifact(
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load zero-pickle JSON model artifact (and any companion `fold_*.lgb` files) from disk."""
    art_file = Path(artifact_path)
    bundle = dict(json.loads(art_file.read_text(encoding="utf-8")))
    for state in bundle.get("fold_states", []):
        fid = int(state.get("fold_id", 0))
        m_dict = state.setdefault("model", {})
        lgb_path = art_file.parent / f"fold_{fid}.lgb"
        if lgb_path.is_file() and not m_dict.get("lgbm_model_str"):
            m_dict["lgbm_model_str"] = lgb_path.read_text(encoding="utf-8")
        xgb_path = art_file.parent / f"fold_{fid}.xgb.json"
        if xgb_path.is_file() and not m_dict.get("xgb_model_json"):
            m_dict["xgb_model_json"] = xgb_path.read_text(encoding="utf-8")
        cbm_path = art_file.parent / f"fold_{fid}.cbm.json"
        if cbm_path.is_file() and not m_dict.get("catboost_model_json"):
            m_dict["catboost_model_json"] = cbm_path.read_text(encoding="utf-8")
    return bundle


def run_inference_pipeline(
    fitted_bundle: dict[str, Any],
    test_df: pd.DataFrame | pl.DataFrame,
) -> np.ndarray:
    """Score unseen `test_df` by averaging calibrated predictions across all CV folds."""
    pdf_test = _to_pandas(test_df)
    feature_cols = list(fitted_bundle["feature_cols"])
    for col in feature_cols:
        if col not in pdf_test.columns:
            pdf_test[col] = np.nan
    pdf_features = pdf_test[feature_cols].reset_index(drop=True)

    fold_preds: list[np.ndarray] = []
    for state in fitted_bundle["fold_states"]:
        clean_df = preprocessing(pdf_features, state["preprocess"])
        feat_df = feature_engineering(clean_df)
        raw = predict_model_stage(feat_df, state["model"])
        y_ref = np.asarray(state["y_tr_ref"], dtype=np.float64)
        fold_preds.append(run_postprocess_stage(raw, y_ref))
    preds = np.asarray(np.mean(np.vstack(fold_preds), axis=0), dtype=np.float64)
    return np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)


def format_submission_frame(
    test_df: pl.DataFrame,
    preds: np.ndarray,
    sample_sub_df: pl.DataFrame | None = None,
    target_col: str = "utility_agent1",
) -> pl.DataFrame:
    """Align predictions to `sample_submission.csv` schema or `[id_col, prediction]`."""
    if sample_sub_df is not None and sample_sub_df.width >= 2:
        pred_col = sample_sub_df.columns[1]
        return sample_sub_df.with_columns(pl.Series(name=pred_col, values=preds))
    id_col = test_df.columns[0] if test_df.columns else "id"
    is_ranking = (
        target_col == id_col
        or bool(CONFIGURED_EVENT_COL)
        or "RMSE".strip().lower() == "c_index"
    )
    out_col = "prediction" if is_ranking else target_col
    return pl.DataFrame({id_col: test_df[id_col], out_col: preds})


def main() -> None:
    parser = argparse.ArgumentParser(description="sovLoop Modular Inference Orchestrator")
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to zero-pickle `model_artifact.json` produced by `training_pipeline.py`.",
    )
    parser.add_argument("--train", default="", help="Optional training CSV fallback.")
    parser.add_argument("--test", required=True, help="Unseen test CSV/Parquet to score.")
    parser.add_argument("--sample-sub", default="", help="Optional sample_submission.csv path.")
    parser.add_argument("--target", default="utility_agent1", help="Target column.")
    parser.add_argument("--output", default="submission.csv", help="Output predictions CSV.")
    args = parser.parse_args()

    art_path = Path(args.artifact)
    if art_path.is_file():
        fitted = load_trained_artifact(art_path)
    elif args.train:
        tr_df = _load_frame(args.train)
        fitted = run_training_pipeline(tr_df, target_col=args.target)
        save_trained_artifact(fitted, art_path)
    else:
        raise FileNotFoundError(
            f"Model artifact not found at {art_path}. Run `training_pipeline.py` first "
            "or supply `--train path/to/train.csv`."
        )

    te_df = _load_frame(args.test)
    has_sub = bool(args.sample_sub) and Path(args.sample_sub).is_file()
    sub_tpl = _load_frame(args.sample_sub) if has_sub else None
    preds = run_inference_pipeline(fitted, te_df)
    out_df = format_submission_frame(te_df, preds, sample_sub_df=sub_tpl, target_col=args.target)
    out_df.write_csv(args.output)
    print(f"Wrote {len(preds)} predictions -> {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
