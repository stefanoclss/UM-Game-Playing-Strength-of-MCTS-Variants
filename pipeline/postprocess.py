"""Stage 5 · Prediction Calibration & Post-Processing (`pipeline/postprocess.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: identity_passthrough
"""

from __future__ import annotations

import numpy as np

ACTIVE_POSTPROCESS_OPS: tuple[str, ...] = ('identity_passthrough',)


def _pav_monotonic_curve(preds: np.ndarray, y_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 17-knot Pool-Adjacent-Violators monotonic calibration curve."""
    order = np.argsort(preds, kind="mergesort")
    x_s = preds[order].astype(np.float64)
    vals: list[float] = []
    wts: list[int] = []
    for y_i in y_ref[order]:
        vals.append(float(y_i))
        wts.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            w_sum = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / float(w_sum)
            wts[-2] = w_sum
            vals.pop()
            wts.pop()
    y_iso = np.repeat(np.asarray(vals, dtype=np.float64), wts)
    q = np.linspace(0.0, 1.0, 17)
    xk = np.quantile(x_s, q)
    yk = np.maximum.accumulate(np.quantile(y_iso, q))
    return xk, yk


def run_postprocess_stage(preds: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Calibrate fold predictions using strictly training-fold target moments."""
    out = preds.astype(np.float64, copy=True)
    if (
        "isotonic_pav_calibration" in ACTIVE_POSTPROCESS_OPS
        and len(out) >= 16
        and len(y_tr) == len(out)
    ):
        xk, yk = _pav_monotonic_curve(out, y_tr)
        out = 0.65 * out + 0.35 * np.interp(out, xk, yk)
    if "rank_calibration_blend" in ACTIVE_POSTPROCESS_OPS and len(out) >= 4:
        ranks = np.argsort(np.argsort(out, kind="mergesort"), kind="mergesort").astype(np.float64)
        pct = np.clip((ranks + 0.5) / float(len(out)), 1e-4, 1.0 - 1e-4)
        rg = 0.6266 * np.log(pct / (1.0 - pct))
        mu, sd = float(np.mean(out)), max(float(np.std(out)), 1e-5)
        out = mu + sd * (0.60 * ((out - mu) / sd) + 0.40 * rg)
    if "tail_shrinkage" in ACTIVE_POSTPROCESS_OPS and len(y_tr) > 0:
        out = 0.94 * out + 0.06 * float(np.mean(y_tr))
    if "clip_training_bounds" in ACTIVE_POSTPROCESS_OPS and len(y_tr) > 0:
        out = np.clip(out, float(np.min(y_tr)), float(np.max(y_tr)))
    return out
