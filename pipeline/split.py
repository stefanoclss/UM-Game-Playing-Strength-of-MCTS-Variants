"""Deterministic Cross-Validation Split Gatekeeper (`pipeline/split.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Strategy: kfold_5_stratified
"""

from __future__ import annotations

import numpy as np

ACTIVE_SPLIT_OPS: tuple[str, ...] = ('kfold_5_stratified',)


def assign_validation_folds(y: np.ndarray, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign deterministic stratified/quantile K-fold indices in `[0, n_folds - 1]`."""
    rng = np.random.default_rng(seed)
    order = np.argsort(y + rng.normal(0.0, 1e-9, size=len(y)), kind="mergesort")
    folds = np.empty(len(y), dtype=np.int32)
    folds[order] = np.arange(len(y), dtype=np.int32) % max(2, int(n_folds))
    return folds
