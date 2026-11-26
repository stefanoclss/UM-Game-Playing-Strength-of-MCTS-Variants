"""Stage 1 · Fold-Safe Data Hygiene & Preprocessing (`pipeline/preprocess.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: missing_indicators
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ACTIVE_PREPROCESS_OPS: tuple[str, ...] = ('missing_indicators',)
SCHEMA_CATEGORICAL_SAMPLE: list[str] = []
SCHEMA_NUMERIC_SAMPLE: list[str] = ['perspective_sign', 'signed_adv_AdvantageP1', 'a1_full', 'a2_full', 'a1_cfg_cat_1', 'a2_cfg_cat_1', 'same_cfg_cat_1', 'a1_cfg_num_2', 'a2_cfg_num_2', 'diff_cfg_num_2', 'sum_cfg_num_2', 'log_ratio_cfg_num_2']


def fit_preprocess_stage(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit imputation statistics, IQR bounds, and category maps strictly on training rows."""
    use_median = "median_impute" in ACTIVE_PREPROCESS_OPS
    cat_cols: list[str] = []
    num_cols: list[str] = []
    cat_maps: dict[str, dict[str, int]] = {}
    cat_freq_maps: dict[str, dict[str, float]] = {}
    fill_values: dict[str, float] = {}
    iqr_bounds: dict[str, list[float]] = {}
    robust_stats: dict[str, list[float]] = {}
    na_cols: list[str] = []
    oof_te_maps: dict[str, dict[str, float]] = {}
    prior = float(np.mean(y_tr)) if y_tr is not None and len(y_tr) > 0 else 0.0
    n_tr = max(len(df_tr), 1)

    for col in df_tr.columns:
        cname = str(col)
        ser = df_tr[col]
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser):
            num_cols.append(cname)
            vals = pd.to_numeric(ser, errors="coerce")
            if bool(vals.isna().any()):
                na_cols.append(cname)
            clean = vals.dropna().to_numpy(dtype=np.float64)
            agg_fn = np.median if use_median else np.mean
            stat = float(agg_fn(clean)) if len(clean) > 0 else 0.0
            fill_values[cname] = stat if np.isfinite(stat) else 0.0
            if len(clean) >= 4:
                q25, q50, q75 = np.percentile(clean, [25.0, 50.0, 75.0])
                iqr = max(float(q75 - q25), 1e-5)
                iqr_bounds[cname] = [float(q25 - 2.5 * iqr), float(q75 + 2.5 * iqr)]
                robust_stats[cname] = [float(q50), float(iqr)]
        else:
            cat_cols.append(cname)
            s_str = ser.astype("object").where(ser.notna(), "__MISSING__").astype(str)
            vc = s_str.value_counts(dropna=False)
            uniq = sorted(set(s_str.tolist()))
            cat_maps[cname] = {val: idx + 1 for idx, val in enumerate(uniq)}
            cat_freq_maps[cname] = {
                str(k): (
                    float(np.log1p(100.0 * float(c) / n_tr))
                    if (float(c) / n_tr) >= 0.02
                    else 0.0
                )
                for k, c in vc.to_dict().items()
            }
            do_te = "oof_target_encoding" in ACTIVE_PREPROCESS_OPS
            if do_te and y_tr is not None and len(y_tr) == len(df_tr):
                tmp = pd.DataFrame({"k": s_str.to_numpy(), "y": y_tr})
                grp = tmp.groupby("k", sort=False)["y"].agg(["count", "mean"])
                smoothed = (grp["count"] * grp["mean"] + 16.0 * prior) / (grp["count"] + 16.0)
                oof_te_maps[cname] = {str(k): float(v) for k, v in smoothed.to_dict().items()}

    return {
        "ops": list(ACTIVE_PREPROCESS_OPS),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": cat_maps,
        "cat_freq_maps": cat_freq_maps,
        "fill_values": fill_values,
        "iqr_bounds": iqr_bounds,
        "robust_stats": robust_stats,
        "na_cols": na_cols[:24],
        "oof_te_maps": oof_te_maps,
        "oof_te_prior": prior,
    }


def preprocessing(df: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Apply one clear column-by-column transformation per step using fold-fitted `state`."""
    out = pd.DataFrame(index=df.index)
    ops = set(state.get("ops", ACTIVE_PREPROCESS_OPS))

    # 1. Encode categorical columns to integer category codes (0 = unseen category)
    for col in state["cat_cols"]:
        s_raw = df[col] if col in df.columns else pd.Series("__MISSING__", index=df.index)
        s_str = s_raw.astype("object").where(s_raw.notna(), "__MISSING__").astype(str)
        out[col] = s_str.map(state["cat_maps"].get(col, {})).fillna(0).astype(np.float64)
        if "oof_target_encoding" in ops and col in state.get("oof_te_maps", {}):
            prior = float(state.get("oof_te_prior", 0.0))
            te_map = state["oof_te_maps"][col]
            out[f"{col}_target_enc"] = s_str.map(te_map).fillna(prior).astype(np.float64)
        if "rare_category_grouping" in ops and col in state.get("cat_freq_maps", {}):
            f_map = state["cat_freq_maps"][col]
            out[f"{col}_freq"] = s_str.map(f_map).fillna(0.0).astype(np.float64)

    # 2. Record missingness indicators before imputing numeric columns
    if "missing_indicators" in ops:
        for col in state.get("na_cols", []):
            out[f"{col}_isna"] = (df[col].isna() if col in df.columns else 1.0)
            out[f"{col}_isna"] = out[f"{col}_isna"].astype(np.float64)
        present_num = [c for c in state["num_cols"] if c in df.columns]
        if present_num:
            out["row_missing_count"] = df[present_num].isna().sum(axis=1).astype(np.float64)

    # 3. Impute numeric columns with training-fold median/mean and clip IQR outliers
    for col in state["num_cols"]:
        vals = (
            pd.to_numeric(df[col], errors="coerce")
            if col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        filled = vals.fillna(float(state["fill_values"].get(col, 0.0))).astype(np.float64)
        if "winsorize_iqr" in ops and col in state.get("iqr_bounds", {}):
            low, high = state["iqr_bounds"][col]
            filled = filled.clip(float(low), float(high))
        if "robust_zscore" in ops and col in state.get("robust_stats", {}):
            med, iqr = state["robust_stats"][col]
            rz = (filled - float(med)) / max(float(iqr), 1e-5)
            out[f"{col}_robz"] = rz.clip(-10.0, 10.0)
        out[col] = filled

    return out


def transform_preprocess_stage(df: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return preprocessing(df, state)
