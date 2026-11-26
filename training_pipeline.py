"""Top-Level Training & K-Fold CV Orchestrator (`training_pipeline.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Executes `preprocessing(df)` -> `feature_engineering(df)` -> `configure_objective_stage`
-> LightGBM/Optuna `fit_model_stage` across strict K-Fold splits, exporting zero-pickle
`artifacts/model_artifact.json` and `artifacts/fold_*.lgb`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from pipeline.augmentation import run_augmentation_stage
from pipeline.eda import profile_dataset
from pipeline.features import feature_engineering
from pipeline.models import fit_model_stage, predict_model_stage
from pipeline.objective import configure_objective_stage
from pipeline.postprocess import run_postprocess_stage
from pipeline.preprocess import fit_preprocess_stage, preprocessing
from pipeline.split import assign_validation_folds

DEFAULT_ARTIFACT_PATH = Path("artifacts/model_artifact.json")
CONFIGURED_EVENT_COL = "None"


def _jsonify(obj: Any) -> Any:
    """Recursively convert NumPy arrays & scalars into pure JSON-safe primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj


def _to_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    if isinstance(df, pl.DataFrame):
        return pd.DataFrame(df.to_dict(as_series=False))
    return df.copy()


def _resolve_event_column(pdf: pd.DataFrame, target_col: str) -> str | None:
    has_ev = bool(CONFIGURED_EVENT_COL) and CONFIGURED_EVENT_COL in pdf.columns
    if has_ev and CONFIGURED_EVENT_COL != target_col:
        return CONFIGURED_EVENT_COL
    return None


def _concordance_index(times: np.ndarray, risk_scores: np.ndarray, events: np.ndarray) -> float:
    n = len(times)
    if n < 2:
        return 0.5
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n, 2500), replace=False) if n > 2500 else np.arange(n)
    t, r, e = times[idx], risk_scores[idx], events[idx]
    valid = (e[:, None] > 0.5) & (t[:, None] < t[None, :])
    denom = float(np.sum(valid))
    if denom <= 0.0:
        return 0.5
    conc = np.sum(valid & (r[:, None] > r[None, :])) + 0.5 * np.sum(
        valid & (r[:, None] == r[None, :])
    )
    return float(conc / denom)


def run_training_pipeline(
    train_df: pd.DataFrame | pl.DataFrame,
    target_col: str = "utility_agent1",
    n_folds: int = 5,
    exclude_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Fit all `pipeline.*` stages across leak-free K-Fold splits and return OOF & states."""
    pdf = _to_pandas(train_df)
    eda_summary = profile_dataset(pdf, target_col=target_col)
    event_col = _resolve_event_column(pdf, target_col)
    first_col = str(pdf.columns[0]) if len(pdf.columns) > 0 else ""
    id_candidates = {first_col} if first_col.lower() in ("id", "row_id", "patient_id") else set()
    blocked = {target_col, *(exclude_cols or []), *id_candidates}
    if event_col:
        blocked.add(event_col)
    feature_cols = [str(c) for c in pdf.columns if str(c) not in blocked]
    if not feature_cols:
        feature_cols = [str(c) for c in pdf.columns if str(c) != target_col]

    y_raw = pd.to_numeric(pdf[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev_arr = (
        pd.to_numeric(pdf[event_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if event_col
        else None
    )
    folds = assign_validation_folds(y_raw, n_folds=n_folds, seed=42)

    oof_preds = np.zeros(len(y_raw), dtype=np.float64)
    fold_states: list[dict[str, Any]] = []
    for fold_id in range(max(2, int(n_folds))):
        va_mask = folds == fold_id
        tr_mask = ~va_mask
        df_tr_raw = pdf.loc[tr_mask, feature_cols].reset_index(drop=True)
        df_va_raw = pdf.loc[va_mask, feature_cols].reset_index(drop=True)

        ev_tr = ev_arr[tr_mask] if ev_arr is not None else None
        y_fold_target, base_sw = configure_objective_stage(y_raw[tr_mask], event_tr=ev_tr)

        pre_state = fit_preprocess_stage(df_tr_raw, y_fold_target)
        df_tr_feat = feature_engineering(preprocessing(df_tr_raw, pre_state))
        df_va_feat = feature_engineering(preprocessing(df_va_raw, pre_state))

        df_aug, y_aug = run_augmentation_stage(df_tr_feat, y_fold_target, seed=42 + fold_id)
        sw_aug = (
            base_sw
            if len(df_aug) == len(y_fold_target)
            else np.ones(len(df_aug), dtype=np.float64)
        )
        model_state = fit_model_stage(
            df_aug,
            y_aug,
            sw_aug,
            cat_cols=pre_state["cat_cols"],
            seed=100 + fold_id,
        )
        raw_va = predict_model_stage(df_va_feat, model_state)
        y_ref = (
            np.quantile(y_fold_target, np.linspace(0.0, 1.0, 65))
            if len(y_fold_target) > 65
            else y_fold_target
        )
        oof_preds[va_mask] = run_postprocess_stage(raw_va, y_ref)
        fold_states.append(
            {
                "fold_id": fold_id,
                "preprocess": pre_state,
                "model": model_state,
                "y_tr_ref": y_ref,
            }
        )

    rmse = float(np.sqrt(np.mean(np.square(y_raw - oof_preds))))
    oof_c_index = _concordance_index(y_raw, oof_preds, ev_arr) if ev_arr is not None else None
    return {
        "eda": eda_summary,
        "target_col": target_col,
        "event_col": event_col,
        "feature_cols": feature_cols,
        "folds": folds,
        "oof_predictions": oof_preds,
        "fold_states": fold_states,
        "oof_rmse": rmse,
        "oof_c_index": oof_c_index,
    }


def save_trained_artifact(
    fitted_bundle: dict[str, Any],
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Save zero-pickle `model_artifact.json` plus individual `fold_{k}.lgb` model files."""
    out_path = Path(artifact_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fstate in fitted_bundle["fold_states"]:
        fid = int(fstate.get("fold_id", 0))
        m_dict = fstate.get("model", {})
        lgb_txt = m_dict.get("lgbm_model_str")
        if isinstance(lgb_txt, str) and lgb_txt.strip():
            (out_path.parent / f"fold_{fid}.lgb").write_bytes(lgb_txt.encode("utf-8"))
        xgb_txt = m_dict.get("xgb_model_json")
        if isinstance(xgb_txt, str) and xgb_txt.strip():
            (out_path.parent / f"fold_{fid}.xgb.json").write_bytes(xgb_txt.encode("utf-8"))
        cbm_txt = m_dict.get("catboost_model_json")
        if isinstance(cbm_txt, str) and cbm_txt.strip():
            (out_path.parent / f"fold_{fid}.cbm.json").write_bytes(cbm_txt.encode("utf-8"))

    payload = {
        "schema_version": "2.0",
        "synthesis_model": "qwen3.5:9b",
        "target_col": fitted_bundle.get("target_col", "utility_agent1"),
        "event_col": fitted_bundle.get("event_col"),
        "feature_cols": fitted_bundle["feature_cols"],
        "fold_states": _jsonify(fitted_bundle["fold_states"]),
        "oof_rmse": float(fitted_bundle["oof_rmse"]),
        "oof_c_index": fitted_bundle.get("oof_c_index"),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="sovLoop Modular Training Orchestrator")
    parser.add_argument("--train", required=True, help="Training CSV or Parquet path.")
    parser.add_argument("--target", default="utility_agent1", help="Target column.")
    parser.add_argument(
        "--artifact-out",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Output zero-pickle JSON model artifact path.",
    )
    parser.add_argument("--oof-out", default="oof_predictions.csv", help="Output OOF CSV.")
    args = parser.parse_args()
    t_path = Path(args.train)
    df = pd.read_parquet(t_path) if t_path.suffix.lower() == ".parquet" else pd.read_csv(t_path)
    result = run_training_pipeline(df, target_col=args.target)
    art_file = save_trained_artifact(result, args.artifact_out)
    pl.DataFrame({"oof_prediction": result["oof_predictions"]}).write_csv(args.oof_out)
    metric_msg = (
        f"OOF C_INDEX={result['oof_c_index']:.5f}"
        if result.get("oof_c_index") is not None
        else f"OOF RMSE={result['oof_rmse']:.5f}"
    )
    print(f"Completed {len(result['fold_states'])}-fold CV | {metric_msg} -> {art_file}")


if __name__ == "__main__":
    main()
