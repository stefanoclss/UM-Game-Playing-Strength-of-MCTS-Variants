"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: log1p_skewed, row_summary_stats, top_pair_products
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('log1p_skewed', 'row_summary_stats', 'top_pair_products')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['log_ratio_cfg_num_2', 'MoveDistanceAverage', 'MoveDistanceMedian', 'txt0_svd_2', 'DecisionFactorAverage', 'diff_cfg_num_2']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    # Signed log1p compression for skewed continuous predictors
    if 'log_ratio_cfg_num_2' in out.columns: out['log_ratio_cfg_num_2_log1p'] = np.sign(out['log_ratio_cfg_num_2']) * np.log1p(np.abs(out['log_ratio_cfg_num_2']))
    if 'MoveDistanceAverage' in out.columns: out['MoveDistanceAverage_log1p'] = np.sign(out['MoveDistanceAverage']) * np.log1p(np.abs(out['MoveDistanceAverage']))
    if 'MoveDistanceMedian' in out.columns: out['MoveDistanceMedian_log1p'] = np.sign(out['MoveDistanceMedian']) * np.log1p(np.abs(out['MoveDistanceMedian']))
    for _c in fallback_cols[:3]:
        if f'{_c}_log1p' not in out.columns:
            out[f'{_c}_log1p'] = np.sign(out[_c]) * np.log1p(np.abs(out[_c]))
    # Multiplicative interaction cross terms
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        out[f'{_a}_mul_{_b}'] = (out[_a] * out[_b]).clip(-25.0, 25.0)
    # Row-wise summary moments across predictors
    if fallback_cols:
        out['row_numeric_mean'] = out[fallback_cols].mean(axis=1)
        out['row_numeric_std'] = out[fallback_cols].std(axis=1).fillna(0.0)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
