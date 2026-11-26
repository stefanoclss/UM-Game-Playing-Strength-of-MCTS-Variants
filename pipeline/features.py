"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: log1p_skewed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('log1p_skewed',)
PRIORITY_SIGNAL_COLUMNS: list[str] = ['a2_full', 'a1_full', 'signed_adv_AdvantageP1', 'AdvantageP1', 'a2_cfg_cat_1', 'a1_cfg_cat_1']


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
    if 'a2_full' in out.columns: out['a2_full_log1p'] = np.sign(out['a2_full']) * np.log1p(np.abs(out['a2_full']))
    if 'a1_full' in out.columns: out['a1_full_log1p'] = np.sign(out['a1_full']) * np.log1p(np.abs(out['a1_full']))
    if 'signed_adv_AdvantageP1' in out.columns: out['signed_adv_AdvantageP1_log1p'] = np.sign(out['signed_adv_AdvantageP1']) * np.log1p(np.abs(out['signed_adv_AdvantageP1']))
    for _c in fallback_cols[:3]:
        if f'{_c}_log1p' not in out.columns:
            out[f'{_c}_log1p'] = np.sign(out[_c]) * np.log1p(np.abs(out[_c]))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
