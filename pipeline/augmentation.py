"""Training-Fold-Only Regularization & Augmentation (`pipeline/augmentation.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: identity_augmentation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_AUGMENTATION_OPS: tuple[str, ...] = ('identity_augmentation',)


def run_augmentation_stage(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Perturb training-fold observations only; validation folds remain 100% untouched."""
    if "gaussian_feature_jitter" not in ACTIVE_AUGMENTATION_OPS or len(df_tr) < 16:
        return df_tr, y_tr
    rng = np.random.default_rng(seed)
    n_extra = max(4, len(df_tr) // 4)
    idx = rng.choice(len(df_tr), size=n_extra, replace=False)
    extra = df_tr.iloc[idx].copy()
    for col in extra.columns:
        extra[col] = extra[col] + rng.normal(0.0, 0.02, size=n_extra)
    return pd.concat([df_tr, extra], ignore_index=True), np.concatenate([y_tr, y_tr[idx]])
