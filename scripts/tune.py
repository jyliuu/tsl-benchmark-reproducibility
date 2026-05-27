#!/usr/bin/env python3
"""
Tune one (dataset, model) pair with the search space from the paper appendix.

Usage:

    python scripts/tune.py --model TSL_R2 --task-id 361234 --n-outer-trials 5

For each outer trial this writes a ``run_seed<seed>_<dataset>_<model>.json`` and
an Optuna JournalStorage ``run...seed<seed>.log`` into the appropriate
``results/`` subdirectory, in the exact layout consumed by
``scripts/regenerate_summary.py``.

Protocol (matches paper appendix, with repeated random splits):
  - Repeat the outer 80/20 split N times; split seeds are range(N) = [0..N-1]
    (``--n-outer-trials N``; default N=1, i.e. one tuning session on seed 0)
  - Per outer trial: an independent Optuna TPE study with 200 trials
  - 10-fold CV on that split's train set (minimising MSE)
  - Best config is refit on the full train split, evaluated once on that test
  - Result files are never deleted, so interrupted runs resume per seed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Per-model: (output category, variant directory, model class name shown in
# JSON, Python factory). The factory takes a params dict and returns a fitted-
# style sklearn estimator.
# ---------------------------------------------------------------------------

# Note: TSL (1-product) — the ``TSLRegressorOneTensor`` ablation in the paper —
# is omitted from the runnable set below because its Python package is not
# publicly released at this time. The precomputed tuning artifacts for that
# row remain in ``results/interpretable/tsl_1product/`` and are picked up by
# ``regenerate_summary.py``; only re-tuning from scratch is unavailable here.

CATEGORY_DIR: Dict[str, Tuple[str, str, str]] = {
    "TSL_R2":        ("interpretable", "tsl_r2",        "TSLRegressor"),
    "TSL_R10":       ("interpretable", "tsl_r10",       "TSLRegressor"),
    "SepALS_r2":     ("interpretable", "sepals_r2",     "SeparatedALSRegressor"),
    "SepALS_r10":    ("interpretable", "sepals_r10",    "SeparatedALSRegressor"),
    "EBM":           ("interpretable", "others",        "ExplainableBoostingRegressor"),
    "LGBM_interp":   ("interpretable", "others",        "LGBMRegressor"),
    "XGB_interp":    ("interpretable", "others",        "XGBRegressor"),
    "LGBM_bb":       ("blackbox",      "others",        "LGBMRegressor"),
    "XGB_bb":        ("blackbox",      "others",        "XGBRegressor"),
    "RF_bb":         ("blackbox",      "others",        "RandomForestRegressor"),
}


def make_estimator(model_key: str, params: dict):
    """Instantiate the (unfit) sklearn estimator with the given hyperparameters."""
    if model_key == "EBM":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(random_state=42, **params)
    if model_key in ("LGBM_interp", "LGBM_bb"):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, verbose=-1, **params)
    if model_key in ("XGB_interp", "XGB_bb"):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, verbosity=0, **params)
    if model_key == "RF_bb":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if model_key in ("TSL_R2", "TSL_R10"):
        from tsl_py.sklearn import TSLRegressor
        return TSLRegressor(seed=42, **params)
    if model_key in ("SepALS_r2", "SepALS_r10"):
        from sepals import SeparatedALSRegressor
        return SeparatedALSRegressor(random_state=42, **params)
    raise ValueError(model_key)


# ---------------------------------------------------------------------------
# Hyperparameter sampling — paper-aligned. Bounds are inclusive for suggest_int
# (so the appendix's "[1, 100)" maps to suggest_int(1, 99)).
# ---------------------------------------------------------------------------

def sample_params(model_key: str, trial) -> dict:
    p: dict = {}

    if model_key in ("TSL_R2", "TSL_R10"):
        p["alpha"]                 = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        p["colsample_bytree"]      = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        p["decay"]                 = trial.suggest_float("decay", 0.2, 1.0)
        if model_key == "TSL_R2":         p["epochs"] = trial.suggest_int("epochs", 1, 2)
        elif model_key == "TSL_R10":      p["epochs"] = trial.suggest_int("epochs", 1, 10)
        p["min_interval_samples"]  = trial.suggest_int("min_interval_samples", 1, 99)
        p["n_iter"]                = trial.suggest_int("n_iter", 10, 250)
        p["refinement_strategy"]   = trial.suggest_categorical("refinement_strategy", ["l2", "huber"])
        p["similarity_threshold"]  = trial.suggest_float("similarity_threshold", 0, 1)
        p["split_try"]             = trial.suggest_int("split_try", 2, 19)
        p["update_clamp"]          = trial.suggest_float("update_clamp", 0.5, 35)
        # Fixed
        p["n_trees"]               = 200 if model_key != "TSL_R10" else 500
        p["bagged"]                = True
        p["split_strategy"]        = "random"

    elif model_key == "EBM":
        p["learning_rate"]         = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
        p["max_bins"]              = trial.suggest_int("max_bins", 64, 256)
        p["min_samples_leaf"]      = trial.suggest_int("min_samples_leaf", 10, 200)
        p["max_rounds"]            = trial.suggest_int("max_rounds", 10, 500)
        p["outer_bags"]            = trial.suggest_int("outer_bags", 4, 20)
        p["smoothing_rounds"]      = trial.suggest_int("smoothing_rounds", 0, 500)
        p["interactions"]          = trial.suggest_int("interactions", 0, 20)
        p["max_interaction_bins"]  = trial.suggest_int("max_interaction_bins", 16, 64)

    elif model_key in ("XGB_interp", "XGB_bb"):
        if model_key == "XGB_interp":
            p["n_estimators"]      = trial.suggest_int("n_estimators", 10, 99)
            p["max_depth"]         = trial.suggest_int("max_depth", 1, 2)
            p["subsample"]         = trial.suggest_float("subsample", 0.6, 1.0)
            p["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.3, 0.9)
            p["gamma"]             = trial.suggest_float("gamma", 0.01, 100, log=True)
            p["reg_alpha"]         = trial.suggest_float("reg_alpha", 0.01, 100, log=True)
            p["reg_lambda"]        = trial.suggest_float("reg_lambda", 0, 1)
        else:
            p["n_estimators"]      = trial.suggest_int("n_estimators", 100, 999)
            p["max_depth"]         = trial.suggest_int("max_depth", 3, 9)
            p["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
            p["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
            p["gamma"]             = trial.suggest_float("gamma", 0.001, 5.0, log=True)
            p["reg_alpha"]         = trial.suggest_float("reg_alpha", 0.001, 10.0, log=True)
            p["reg_lambda"]        = trial.suggest_float("reg_lambda", 0, 10.0)
        p["learning_rate"]         = trial.suggest_float("learning_rate", 0.01, 0.3)

    elif model_key in ("LGBM_interp", "LGBM_bb"):
        if model_key == "LGBM_interp":
            p["n_estimators"]      = trial.suggest_int("n_estimators", 10, 99)
            p["num_leaves"]        = trial.suggest_int("num_leaves", 25, 49)
            p["max_depth"]         = trial.suggest_int("max_depth", 1, 2)
            p["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 29)
            p["subsample"]         = trial.suggest_float("subsample", 0.3, 0.8)
            p["reg_alpha"]         = trial.suggest_float("reg_alpha", 0, 0.5)
            p["reg_lambda"]        = trial.suggest_float("reg_lambda", 0, 0.5)
        else:
            p["n_estimators"]      = trial.suggest_int("n_estimators", 100, 999)
            p["num_leaves"]        = trial.suggest_int("num_leaves", 31, 126)
            p["max_depth"]         = trial.suggest_int("max_depth", 3, 9)
            p["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 29)
            p["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
            p["reg_alpha"]         = trial.suggest_float("reg_alpha", 0, 10.0)
            p["reg_lambda"]        = trial.suggest_float("reg_lambda", 0, 10.0)
        p["learning_rate"]         = trial.suggest_float("learning_rate", 0.01, 0.3)

    elif model_key == "RF_bb":
        p["n_estimators"]          = trial.suggest_int("n_estimators", 100, 999)
        p["max_depth"]             = trial.suggest_int("max_depth", 5, 19)
        p["min_samples_split"]     = trial.suggest_int("min_samples_split", 2, 19)
        p["min_samples_leaf"]      = trial.suggest_int("min_samples_leaf", 1, 9)
        p["max_features"]          = trial.suggest_float("max_features", 0.3, 0.8)

    elif model_key in ("SepALS_r2", "SepALS_r10"):
        p["rank"]          = trial.suggest_int("rank", 1, 2 if model_key == "SepALS_r2" else 10)
        p["degree"]        = trial.suggest_int("degree", 2, 10)
        p["basis"]         = trial.suggest_categorical("basis", ["legendre", "monomial"])
        p["penalty_kind"]  = trial.suggest_categorical("penalty_kind", ["degree", "degree2"])
        p["ridge"]         = trial.suggest_float("ridge", 1e-12, 1e-2, log=True)
        p["smoothness"]    = trial.suggest_float("smoothness", 1e-10, 10.0, log=True)
        p["max_sweeps"]    = trial.suggest_int("max_sweeps", 10, 100)
        p["tol"]           = trial.suggest_float("tol", 1e-10, 1e-4, log=True)
        p["n_init"]        = trial.suggest_int("n_init", 1, 5)
        p["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        p["refit_scales"]  = True

    else:
        raise ValueError(model_key)
    return p


# ---------------------------------------------------------------------------
# Dataset loading: minimal version of the cluster-script pipeline.
# 80/20 train/test, TargetEncoder for categoricals.
# ---------------------------------------------------------------------------

def load_task(task_id: int):
    import openml
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from category_encoders import TargetEncoder

    task = openml.tasks.get_task(task_id)
    ds = task.get_dataset()
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):    y = pd.Series(y)

    cats = list(X.select_dtypes(include=["category", "object"]).columns)
    nums = list(X.select_dtypes(include=["number"]).columns)
    bools = list(X.select_dtypes(include=["bool"]).columns)
    transformers = []
    if nums:  transformers.append(("num", "passthrough", nums))
    if bools: transformers.append(("bool", "passthrough", bools))
    if cats:  transformers.append(("cat", TargetEncoder(), cats))
    X_proc = ColumnTransformer(transformers=transformers).fit_transform(X, y)
    if hasattr(X_proc, "toarray"): X_proc = X_proc.toarray()
    y_arr = y.astype(float).values

    return ds.name, X_proc, y_arr


# ---------------------------------------------------------------------------
# Tuning driver
# ---------------------------------------------------------------------------

def _swallow_rust_panics(objective):
    """Convert Rust panics (BaseException, not Exception) to ValueError so Optuna's
    catch=(ValueError,) marks the trial failed instead of aborting the study."""
    def wrapped(trial):
        try:
            return objective(trial)
        except Exception:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            raise ValueError(f"wrapped {type(e).__name__}: {e}") from e
    return wrapped


def run_seed(model_key: str, task_id: int, seed: int, output_root: Path,
             dataset_name: str, X_proc, y_arr, n_trials: int = 200) -> dict:
    """Run ONE (model, dataset, seed) outer trial.

    Splits the preprocessed data 80/20 with ``random_state=seed``, runs a full,
    resumable Optuna study (its own per-seed ``.log``), refits the best config on
    the train split, evaluates once on the test split, and writes
    ``run_seed{seed}_{dataset}_{model}.json``. Never deletes any prior file, so an
    interrupted run resumes: a finished seed skips HPO, a half-done seed resumes
    mid-search. Returns the written record dict.
    """
    import optuna
    from optuna.storages.journal import JournalFileBackend
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error

    category, variant, model_class_name = CATEGORY_DIR[model_key]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = output_root / category / variant / dataset_name / model_class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_proc, y_arr, test_size=0.2, random_state=seed)

    log_path = out_dir / f"{dataset_name}_{model_class_name}_seed{seed}.log"
    storage = optuna.storages.JournalStorage(JournalFileBackend(str(log_path)))
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=f"{model_class_name}_seed{seed}",
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    def objective(trial):
        params = sample_params(model_key, trial)
        est = make_estimator(model_key, params)
        scores = cross_val_score(est, X_tr, y_tr, cv=10, scoring="neg_mean_squared_error",
                                 n_jobs=-1, error_score="raise")
        return -float(np.mean(scores))

    remaining = n_trials - len(study.trials)
    if remaining > 0:
        start = dt.datetime.now()
        study.optimize(_swallow_rust_panics(objective), n_trials=remaining,
                       catch=(ValueError,))
        print(f"  seed={seed} optimized in {(dt.datetime.now() - start)}", flush=True)

    # Refit best on full train, evaluate on this seed's test split.
    best = study.best_params
    est = make_estimator(model_key, sample_params_from_best(model_key, best))
    est.fit(X_tr, y_tr)
    pred = est.predict(X_te)
    test_mse = float(mean_squared_error(y_te, pred))
    test_rmse = float(np.sqrt(test_mse))

    record = {
        "success": True,
        "dataset": dataset_name,
        "model": model_class_name,
        "split_seed": seed,
        "dataset_config": {"type": "openml_task", "task_id": task_id, "name": dataset_name},
        "n_folds": 1,
        "mean_test_rmse": test_rmse,
        "results": [{
            "model": model_class_name,
            "best_params": best,
            "fixed_params": {},
            "best_cv_score": float(study.best_value) if study.best_value is not None else None,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "dataset": dataset_name,
        }],
        "start_timestamp": None,
        "end_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out_path = out_dir / f"run_seed{seed}_{dataset_name}_{model_class_name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"  wrote {out_path}  (seed={seed}, test RMSE = {test_rmse:.4f})", flush=True)
    return record


def tune(model_key: str, task_id: int, output_root: Path, n_trials: int = 200,
         n_outer_trials: int = 1) -> None:
    """Local (non-cluster) driver: load the dataset once, then run each outer
    trial sequentially. For massive parallelism, use the cluster scripts, which
    dispatch one ``run_seed`` per (model, dataset, seed) as a separate job."""
    dataset_name, X_proc, y_arr = load_task(task_id)
    for seed in range(n_outer_trials):
        run_seed(model_key, task_id, seed, output_root, dataset_name, X_proc, y_arr,
                 n_trials=n_trials)


def sample_params_from_best(model_key: str, best: dict) -> dict:
    """Re-derive the full params dict (including fixed components like n_trees, basis-conditional
    penalty_kind for SepALS) from the Optuna-stored best_params."""
    out = dict(best)
    if model_key in ("TSL_R2", "TSL_R10"):
        out["n_trees"]        = 200 if model_key != "TSL_R10" else 500
        out["bagged"]         = True
        out["split_strategy"] = "random"
    elif model_key in ("SepALS_r2", "SepALS_r10"):
        out["refit_scales"] = True
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    results_default = here.parent / "results"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(CATEGORY_DIR))
    p.add_argument("--task-id", type=int, required=True,
                   help="OpenML task ID (e.g. 361266 for kings_county; see CTR23_DATASETS in regenerate_summary.py)")
    p.add_argument("--results-root", type=Path, default=results_default)
    p.add_argument("--n-trials", type=int, default=200,
                   help="Optuna HPO trials per outer trial (inner search budget).")
    p.add_argument("--n-outer-trials", type=int, default=1, metavar="N",
                   help="Number of repeated 80/20 splits. Split seeds are range(N) = [0..N-1]. "
                        "Full HPO runs per outer trial. N=1 (default) is one tuning session (seed 0).")
    args = p.parse_args()

    tune(args.model, args.task_id, args.results_root, n_trials=args.n_trials,
         n_outer_trials=args.n_outer_trials)


if __name__ == "__main__":
    main()
