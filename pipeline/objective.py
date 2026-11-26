"""Stage 3 · Target Formulation & Loss Weighting (`pipeline/objective.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: standard_task_loss
"""

from __future__ import annotations

import numpy as np

ACTIVE_OBJECTIVE_OPS: tuple[str, ...] = ('standard_task_loss',)


def _kaplan_meier_risk_target(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Compute fold-isolated Kaplan-Meier risk target (higher = earlier event / higher risk)."""
    uniq_t = np.unique(times[events > 0.5])
    surv_prob = 1.0
    km_at_t: dict[float, float] = {}
    for t_val in uniq_t:
        at_risk = float(np.sum(times >= t_val))
        d_count = float(np.sum((times == t_val) & (events > 0.5)))
        if at_risk > 0.0:
            surv_prob *= 1.0 - (d_count / at_risk)
        km_at_t[float(t_val)] = surv_prob

    out = np.empty(len(times), dtype=np.float64)
    sorted_t = np.asarray(list(km_at_t.keys()), dtype=np.float64)
    sorted_s = np.asarray(list(km_at_t.values()), dtype=np.float64)
    for i, (t_i, e_i) in enumerate(zip(times, events)):
        idx = int(np.searchsorted(sorted_t, t_i, side="right")) - 1
        s_val = float(sorted_s[idx]) if idx >= 0 and len(sorted_s) > 0 else 1.0
        # Early event has high S(t_i) (~1.0); late event has low S(t_i). Censored rows sit below.
        out[i] = s_val if e_i > 0.5 else (s_val - 0.10)
    return out


def _nelson_aalen_risk_target(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Compute fold-isolated Negative Nelson-Aalen cumulative hazard target `-H_NA(t)`."""
    uniq_t = np.unique(times[events > 0.5])
    cum_h = 0.0
    na_at_t: dict[float, float] = {}
    for t_val in uniq_t:
        at_risk = float(np.sum(times >= t_val))
        d_count = float(np.sum((times == t_val) & (events > 0.5)))
        if at_risk > 0.0:
            cum_h += d_count / at_risk
        na_at_t[float(t_val)] = cum_h

    out = np.empty(len(times), dtype=np.float64)
    sorted_t = np.asarray(list(na_at_t.keys()), dtype=np.float64)
    sorted_h = np.asarray(list(na_at_t.values()), dtype=np.float64)
    for i, (t_i, e_i) in enumerate(zip(times, events)):
        idx = int(np.searchsorted(sorted_t, t_i, side="right")) - 1
        h_val = float(sorted_h[idx]) if idx >= 0 and len(sorted_h) > 0 else 0.0
        out[i] = -h_val if e_i > 0.5 else (-h_val - 0.20)
    return out


def configure_objective_stage(
    y_tr: np.ndarray,
    event_tr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform training-fold targets and compute per-sample weights without leakage."""
    weights = np.ones(len(y_tr), dtype=np.float64)
    y_target = y_tr.astype(np.float64, copy=True)

    if event_tr is not None and "nelson_aalen_hazard_target" in ACTIVE_OBJECTIVE_OPS:
        y_target = _nelson_aalen_risk_target(y_tr, event_tr)
    elif event_tr is not None and (
        "kaplan_meier_target" in ACTIVE_OBJECTIVE_OPS
        or "cox_aft_target" in ACTIVE_OBJECTIVE_OPS
    ):
        y_target = _kaplan_meier_risk_target(y_tr, event_tr)
    elif "log1p_skewed_target" in ACTIVE_OBJECTIVE_OPS:
        if event_tr is not None:
            neg_log = -np.log1p(np.maximum(y_tr, 0.0))
            y_target = np.where(event_tr > 0.5, neg_log, neg_log - 0.38)
        else:
            y_target = np.sign(y_tr) * np.log1p(np.abs(y_tr))
    elif "quantile_rank_target" in ACTIVE_OBJECTIVE_OPS and len(y_tr) >= 4:
        oriented = -y_tr if event_tr is not None else y_tr
        ranks = (np.argsort(np.argsort(oriented, kind="mergesort")) + 0.5) / float(len(y_tr))
        pct = np.clip(ranks, 1e-4, 1.0 - 1e-4)
        rg = 0.6266 * np.log(pct / (1.0 - pct))
        y_target = np.where(event_tr > 0.5, rg, rg - 0.65) if event_tr is not None else rg

    if "huber_robust_objective" in ACTIVE_OBJECTIVE_OPS and len(y_target) > 4:
        dev = np.abs(y_target - float(np.median(y_target)))
        scale = max(float(np.percentile(dev, 75)), 1e-4)
        weights = 1.0 / np.maximum(1.0, dev / (2.0 * scale))
    elif "balanced_class_weights" in ACTIVE_OBJECTIVE_OPS and len(y_target) > 4:
        pos_rate = float(np.clip(np.mean(y_target > 0.5), 0.02, 0.98))
        weights = np.where(y_target > 0.5, 0.5 / pos_rate, 0.5 / (1.0 - pos_rate))

    return y_target, weights
