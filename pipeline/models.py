"""Stage 4 · Multi-Family Production Ensemble & Optuna Tuning (`pipeline/models.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Target: `utility_agent1` (regression | RMSE)
Active Model Operations: lgbm_optuna_tuned, boosted_residual_trees, hybrid_ridge_boost_blend
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

ACTIVE_MODEL_OPS: tuple[str, ...] = ('lgbm_optuna_tuned', 'boosted_residual_trees', 'hybrid_ridge_boost_blend')
IS_BINARY_TASK: bool = "regression" == "binary_classification"


def _to_numeric_matrix(frame: pd.DataFrame, dtype: Any = np.float64) -> np.ndarray:
    """Convert categorical/numeric DataFrame columns into a dense NumPy array."""
    coded = frame.apply(lambda s: s.cat.codes if hasattr(s, "cat") else s)
    return coded.to_numpy(dtype=dtype)


def _fit_ridge_fallback(
    x_mat: np.ndarray,
    y_tr: np.ndarray,
    sw: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit fast closed-form L2-regularized linear baseline."""
    design = np.column_stack([x_mat, np.ones(len(x_mat), dtype=np.float64)])
    reg = alpha * np.eye(design.shape[1], dtype=np.float64)
    reg[-1, -1] = 1e-4
    xw = design * sw[:, None]
    weights = np.linalg.lstsq(design.T @ xw + reg, xw.T @ y_tr, rcond=None)[0]
    return {"backend": "ridge", "weights": weights, "is_binary": IS_BINARY_TASK}


def _tune_lgbm_with_optuna(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    sw_fit: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    seed: int,
) -> dict[str, Any]:
    """Run leak-free inner-fold Optuna TPE hyperparameter optimization for LightGBM."""
    ds_cfg = {"feature_pre_filter": False, "verbosity": -1}
    dtrain = lgb.Dataset(
        x_fit,
        label=y_fit,
        weight=sw_fit,
        categorical_feature=cat_cols or "auto",
        params=ds_cfg,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        x_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=cat_cols or "auto",
        params=ds_cfg,
        free_raw_data=False,
    )
    n_trials = 8 if len(x_fit) >= 500 else 3
    max_rounds = 400 if len(x_fit) >= 500 else 80
    obj_name = "binary" if IS_BINARY_TASK else "regression"

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": obj_name,
            "metric": "binary_logloss" if IS_BINARY_TASK else "rmse",
            "verbosity": -1,
            "feature_pre_filter": False,
            "n_jobs": 4,
            "seed": seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, max(10, min(80, len(x_fit) // 4))
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        preds = np.asarray(bst.predict(x_val), dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(y_val - preds))))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(_objective, n_trials=n_trials, timeout=15.0, show_progress_bar=False)
    return dict(study.best_params)


def _fit_optional_xgb(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    sw_fit: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[str, np.ndarray | None]:
    """Fit histogram XGBoost member when requested in `ACTIVE_MODEL_OPS`."""
    if not any("xgb" in op for op in ACTIVE_MODEL_OPS):
        return "", None
    import xgboost as xgb

    dm_tr = xgb.DMatrix(_to_numeric_matrix(x_fit, np.float32), label=y_fit, weight=sw_fit)
    dm_va = xgb.DMatrix(_to_numeric_matrix(x_val, np.float32), label=y_val)
    xgb_bst = xgb.train(
        {
            "objective": "binary:logistic" if IS_BINARY_TASK else "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "seed": seed,
            "verbosity": 0,
        },
        dm_tr,
        num_boost_round=min(rounds, 320),
        evals=[(dm_va, "val")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    raw_json = xgb_bst.save_raw(raw_format="json").decode("utf-8")
    return raw_json, np.asarray(xgb_bst.predict(dm_va), dtype=np.float64)


def _fit_optional_catboost(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    sw_fit: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[str, np.ndarray | None]:
    """Fit symmetric-oblivious CatBoost member when requested in `ACTIVE_MODEL_OPS`."""
    if not any("catboost" in op for op in ACTIVE_MODEL_OPS):
        return "", None
    import catboost as cb

    xf_cb = _to_numeric_matrix(x_fit, np.float64)
    xv_cb = _to_numeric_matrix(x_val, np.float64)
    cb_kw = {
        "iterations": min(rounds, 320),
        "depth": 6,
        "learning_rate": 0.045,
        "random_seed": seed,
        "verbose": False,
    }
    cb_model = cb.CatBoostClassifier(**cb_kw) if IS_BINARY_TASK else cb.CatBoostRegressor(**cb_kw)
    cb_model.fit(
        xf_cb,
        y_fit,
        sample_weight=sw_fit,
        eval_set=(xv_cb, y_val),
        early_stopping_rounds=25,
        verbose=False,
    )
    with tempfile.TemporaryDirectory() as tmp_d:
        cb_p = Path(tmp_d) / "fold.cbm.json"
        cb_model.save_model(str(cb_p), format="json")
        cbm_json = cb_p.read_text(encoding="utf-8")
    cb_val = cb_model.predict_proba(xv_cb)[:, 1] if IS_BINARY_TASK else cb_model.predict(xv_cb)
    return cbm_json, np.asarray(cb_val, dtype=np.float64)


def _fit_optional_two_stage(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_val: pd.DataFrame,
    valid_cats: list[str],
    rounds: int,
    seed: int,
) -> tuple[str, str, np.ndarray | None]:
    """Fit Two-Stage Incidence + Counterfactual Regressor when active."""
    if "two_stage_counterfactual_gbdt" not in ACTIVE_MODEL_OPS or IS_BINARY_TASK:
        return "", "", None
    inc_y = (y_fit > float(np.median(y_fit))).astype(np.float64)
    if not (0.05 < float(np.mean(inc_y)) < 0.95):
        return "", "", None
    cat_arg: list[str] | str = valid_cats or "auto"
    ds_p = {"verbosity": -1}
    dt_inc = lgb.Dataset(x_fit, label=inc_y, categorical_feature=cat_arg, params=ds_p)
    bst_inc = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "verbosity": -1,
            "seed": seed + 7,
        },
        dt_inc,
        num_boost_round=min(rounds, 220),
    )
    dt_cf = lgb.Dataset(x_fit, label=y_fit, categorical_feature=cat_arg, params={"verbosity": -1})
    bst_cf = lgb.train(
        {
            "objective": "regression_l1",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "verbosity": -1,
            "seed": seed + 13,
        },
        dt_cf,
        num_boost_round=min(rounds, 220),
    )
    p_inc = np.asarray(bst_inc.predict(x_val), dtype=np.float64)
    p_cf = np.asarray(bst_cf.predict(x_val), dtype=np.float64)
    return bst_inc.model_to_string(), bst_cf.model_to_string(), 0.45 * p_inc + 0.55 * p_cf


def _solve_quantile_cdf_blend(
    val_preds_map: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Solve convex 21-knot Quantile-CDF ensemble weights on inner fold holdout."""
    if len(val_preds_map) <= 1:
        only_key = next(iter(val_preds_map.keys()), "lgbm")
        return {only_key: 1.0}, {}
    q_grid = np.linspace(0.0, 1.0, 21)
    q_knots: dict[str, list[float]] = {}
    keys = list(val_preds_map.keys())
    cdfs: list[np.ndarray] = []
    for k in keys:
        vec = np.nan_to_num(val_preds_map[k])
        kn = np.quantile(vec, q_grid)
        q_knots[k] = [float(v) for v in kn]
        cdfs.append(np.interp(vec, kn, q_grid))
    order = np.argsort(np.argsort(y_val, kind="mergesort"))
    y_ranks = (order + 0.5) / max(float(len(y_val)), 1.0)
    mat = np.column_stack(cdfs)
    raw_w = np.linalg.lstsq(mat.T @ mat + 1.5 * np.eye(len(keys)), mat.T @ y_ranks, rcond=None)[0]
    pos_w = np.maximum(raw_w, 0.04)
    pos_w = pos_w / max(float(np.sum(pos_w)), 1e-9)
    return {k: round(float(w), 6) for k, w in zip(keys, pos_w)}, q_knots


def fit_model_stage(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    cat_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit multi-family GBDT + Counterfactual + Ridge ensemble or solo estimator on fold data."""
    sw = sample_weight if sample_weight is not None else np.ones(len(y_tr), dtype=np.float64)
    only_ridge = all(op.startswith("ridge_alpha_") for op in ACTIVE_MODEL_OPS)
    if only_ridge or len(df_tr) < 20:
        is_low = "ridge_alpha_0.1" in ACTIVE_MODEL_OPS
        alpha = 0.1 if is_low else (15.0 if "ridge_alpha_15.0" in ACTIVE_MODEL_OPS else 1.0)
        return _fit_ridge_fallback(df_tr.to_numpy(dtype=np.float64), y_tr, sw, alpha=alpha)

    work_df = df_tr.copy()
    valid_cats = [c for c in (cat_cols or []) if c in work_df.columns]
    for col in valid_cats:
        work_df[col] = work_df[col].astype(np.int32).astype("category")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(work_df))
    n_val = max(8, int(0.15 * len(work_df)))
    va_idx, fit_idx = perm[:n_val], perm[n_val:]
    x_fit, y_fit, sw_fit = work_df.iloc[fit_idx], y_tr[fit_idx], sw[fit_idx]
    x_val, y_val = work_df.iloc[va_idx], y_tr[va_idx]

    tuned_params = (
        _tune_lgbm_with_optuna(x_fit, y_fit, sw_fit, x_val, y_val, valid_cats, seed)
        if any("optuna" in op for op in ACTIVE_MODEL_OPS)
        else {}
    )
    base_params: dict[str, Any] = {
        "objective": "binary" if IS_BINARY_TASK else "regression",
        "metric": "binary_logloss" if IS_BINARY_TASK else "rmse",
        "verbosity": -1,
        "feature_pre_filter": False,
        "n_jobs": 4,
        "seed": seed,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": max(4, min(25, len(x_fit) // 6)),
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_lambda": 1.0,
        **tuned_params,
    }
    ds_cfg = {"feature_pre_filter": False, "verbosity": -1}
    dtrain = lgb.Dataset(
        x_fit,
        label=y_fit,
        weight=sw_fit,
        categorical_feature=valid_cats or "auto",
        params=ds_cfg,
    )
    dvalid = lgb.Dataset(
        x_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=valid_cats or "auto",
        params=ds_cfg,
    )
    rounds = 600 if len(x_fit) >= 500 else 120
    booster = lgb.train(
        base_params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )
    gain_imp = booster.feature_importance(importance_type="gain")
    top_imp = dict(
        sorted(
            zip(work_df.columns.astype(str), [round(float(v), 4) for v in gain_imp]),
            key=lambda kv: kv[1],
            reverse=True,
        )[:15]
    )
    val_preds_map: dict[str, np.ndarray] = {
        "lgbm": np.asarray(booster.predict(x_val), dtype=np.float64)
    }
    xgb_json, xgb_val = _fit_optional_xgb(x_fit, y_fit, sw_fit, x_val, y_val, rounds, seed)
    if xgb_val is not None:
        val_preds_map["xgb"] = xgb_val
    cbm_json, cbm_val = _fit_optional_catboost(x_fit, y_fit, sw_fit, x_val, y_val, rounds, seed)
    if cbm_val is not None:
        val_preds_map["catboost"] = cbm_val
    ts_cls, ts_reg, ts_val = _fit_optional_two_stage(x_fit, y_fit, x_val, valid_cats, rounds, seed)
    if ts_val is not None:
        val_preds_map["two_stage"] = ts_val

    ridge_w: list[float] = []
    if "hybrid_ridge_boost_blend" in ACTIVE_MODEL_OPS:
        xf_mat = _to_numeric_matrix(x_fit, np.float64)
        xv_mat = _to_numeric_matrix(x_val, np.float64)
        r_state = _fit_ridge_fallback(xf_mat, y_fit, sw_fit, alpha=5.0)
        ridge_w = [float(v) for v in r_state["weights"]]
        val_preds_map["ridge"] = np.column_stack([xv_mat, np.ones(len(xv_mat))]) @ np.asarray(
            ridge_w
        )

    blend_weights, q_knots = _solve_quantile_cdf_blend(val_preds_map, y_val)
    return {
        "backend": "multi_ensemble" if len(val_preds_map) > 1 else "lightgbm",
        "ops": list(ACTIVE_MODEL_OPS),
        "cat_cols": valid_cats,
        "feature_names": [str(c) for c in work_df.columns],
        "lgbm_model_str": booster.model_to_string(),
        "xgb_model_json": xgb_json,
        "catboost_model_json": cbm_json,
        "two_stage_cls_str": ts_cls,
        "two_stage_reg_str": ts_reg,
        "ridge_weights": ridge_w,
        "blend_weights": blend_weights,
        "quantile_knots": q_knots,
        "optuna_best_params": tuned_params,
        "feature_importances": top_imp,
        "is_binary": IS_BINARY_TASK,
    }


def predict_model_stage(df_eval: pd.DataFrame, model_state: dict[str, Any]) -> np.ndarray:
    """Generate out-of-fold or holdout predictions from zero-pickle multi-family `model_state`."""
    if model_state.get("backend") in ("lightgbm", "multi_ensemble") and model_state.get(
        "lgbm_model_str"
    ):
        work = df_eval.copy()
        feat_names = model_state.get("feature_names", list(work.columns))
        for col in feat_names:
            if col not in work.columns:
                work[col] = 0.0
        work = work[feat_names]
        for col in model_state.get("cat_cols", []):
            if col in work.columns:
                work[col] = work[col].astype(np.int32).astype("category")

        lgb_bst = lgb.Booster(model_str=str(model_state["lgbm_model_str"]))
        preds_map: dict[str, np.ndarray] = {
            "lgbm": np.asarray(lgb_bst.predict(work), dtype=np.float64)
        }
        num_mat = _to_numeric_matrix(work, np.float64)
        if model_state.get("xgb_model_json"):
            import xgboost as xgb

            xb = xgb.Booster()
            xb.load_model(bytearray(str(model_state["xgb_model_json"]).encode("utf-8")))
            preds_map["xgb"] = np.asarray(
                xb.predict(xgb.DMatrix(num_mat.astype(np.float32))), dtype=np.float64
            )
        if model_state.get("catboost_model_json"):
            import catboost as cb

            is_bin = bool(model_state.get("is_binary"))
            cm = cb.CatBoostClassifier() if is_bin else cb.CatBoostRegressor()
            with tempfile.TemporaryDirectory() as tmp_d:
                cp = Path(tmp_d) / "fold.cbm.json"
                cp.write_text(str(model_state["catboost_model_json"]), encoding="utf-8")
                cm.load_model(str(cp), format="json")
            c_out = cm.predict_proba(num_mat)[:, 1] if is_bin else cm.predict(num_mat)
            preds_map["catboost"] = np.asarray(c_out, dtype=np.float64)
        if model_state.get("two_stage_cls_str") and model_state.get("two_stage_reg_str"):
            bst_i = lgb.Booster(model_str=str(model_state["two_stage_cls_str"]))
            bst_r = lgb.Booster(model_str=str(model_state["two_stage_reg_str"]))
            p_i = np.asarray(bst_i.predict(work), dtype=np.float64)
            p_r = np.asarray(bst_r.predict(work), dtype=np.float64)
            preds_map["two_stage"] = 0.45 * p_i + 0.55 * p_r
        if model_state.get("ridge_weights"):
            rw = np.asarray(model_state["ridge_weights"], dtype=np.float64)
            des = np.column_stack([num_mat, np.ones(len(num_mat), dtype=np.float64)])
            if len(rw) == des.shape[1]:
                preds_map["ridge"] = des @ rw

        bw = model_state.get("blend_weights") or {"lgbm": 1.0}
        knots = model_state.get("quantile_knots") or {}
        if len(preds_map) == 1 or not knots:
            preds = preds_map["lgbm"]
        else:
            q_grid = np.linspace(0.0, 1.0, 21)
            acc = np.zeros(len(work), dtype=np.float64)
            tot = 0.0
            for k, vec in preds_map.items():
                w = float(bw.get(k, 0.0))
                if w <= 0.0:
                    continue
                kn = knots.get(k)
                mapped = (
                    np.interp(np.nan_to_num(vec), np.asarray(kn, dtype=np.float64), q_grid)
                    if kn and len(kn) == 21
                    else vec
                )
                acc += w * mapped
                tot += w
            preds = acc / max(tot, 1e-9)
        return np.clip(preds, 0.0, 1.0) if model_state.get("is_binary") else preds

    x_eval = df_eval.to_numpy(dtype=np.float64)
    design = np.column_stack([x_eval, np.ones(len(x_eval), dtype=np.float64)])
    weights = np.asarray(model_state["weights"], dtype=np.float64)
    raw = np.asarray(design @ weights, dtype=np.float64)
    return np.clip(raw, 0.0, 1.0) if model_state.get("is_binary") else raw
