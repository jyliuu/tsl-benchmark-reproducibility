# TSL Benchmark Reproducibility

This bundle contains the raw hyperparameter-tuning artifacts and the scripts
used to produce the full benchmark table in the appendix of the TSL paper
(OpenML CTR 23 suite, 27 regression datasets, 11 model variants).

```
tsl-benchmark-reproducibility/
├── README.md                              this file
├── requirements.txt                       pip-installable deps
├── summary_interpretable_vs_blackbox.txt  regenerated full benchmark table
├── results/                               raw tuning artifacts (see "Results layout")
│   ├── interpretable/
│   │   ├── tsl_r2/         <dataset>/TSLRegressor/...
│   │   ├── tsl_r10/        <dataset>/TSLRegressor/...
│   │   ├── tsl_1product/   <dataset>/TSLRegressorOneTensor/...
│   │   ├── sepals_r2/      <dataset>/SeparatedALSRegressor/...
│   │   ├── sepals_r10/     <dataset>/SeparatedALSRegressor/...
│   │   └── others/         <dataset>/{ExplainableBoostingRegressor,LGBMRegressor,XGBRegressor}/...
│   └── blackbox/
│       └── others/         <dataset>/{LGBMRegressor,RandomForestRegressor,XGBRegressor}/...
└── scripts/
    ├── regenerate_summary.py              rebuild the table from results/
    ├── tune.py                            tune one (dataset, model) cell
    └── tune_all.sh                        loop tune.py over the 27 × 11 grid
```

## What's inside `results/`

Each `<dataset>/<ModelClass>/` directory contains two files:

- `run_<id>_<dataset>_<ModelClass>.json` — best-trial summary: `mean_test_rmse`
  on the held-out test split, `best_params`, dataset metadata.
- `<dataset>_<ModelClass>.log` — Optuna JournalStorage log (JSONL); contains
  every trial's sampled parameters, value, and state. Re-openable with
  `optuna.storages.JournalStorage(JournalFileBackend(path))` for full replay or
  trial-level analysis.

Excluded from the bundle (regeneratable from `best_params`):

- `*_best.bin` — pickled fitted models (~536 MB total)
- `*_best.sqlite` — TSL visualization databases (~52 GB total)
- The Rust core / Python bindings for TSL and SepALS — distributed separately.

## Reproduce the summary table

No external packages required for this step (Python ≥ 3.10 stdlib only):

```bash
python scripts/regenerate_summary.py
```

This walks `results/`, computes mean test RMSE per `(dataset, model)`, and
writes `summary_interpretable_vs_blackbox.txt` — the table that the appendix's
full-benchmark figure is derived from.

## Re-run the tuning sweep

> **Note.** The Python package backing the TSL (1-product) ablation
> (`TSLRegressorOneTensor`, i.e.\ the `TSL_1product` model key) is not
> publicly released at this time. `tune.py` and `tune_all.sh` therefore
> exclude it from the runnable model set. The precomputed tuning artifacts
> for that row are still shipped in `results/interpretable/tsl_1product/`
> and are picked up by `regenerate_summary.py`, so the regenerated summary
> table includes its column — only re-tuning that one variant from scratch
> is unavailable here.

This re-tunes from scratch (or resumes a partially-complete Optuna study).

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Tune one `(model, dataset)` cell:

    ```bash
    python scripts/tune.py --model XGB_bb --task-id 361266   # kings_county, XGB black-box
    ```

3. Or the entire 27 × 11 grid:

    ```bash
    bash scripts/tune_all.sh                # all 11 models × 27 datasets
    bash scripts/tune_all.sh XGB_bb         # one model across all 27 datasets
    ```

Each invocation:
- Loads the OpenML task, applies an 80/20 train/test split with `random_state=42`.
- Builds the paper-aligned search space inline in `tune.py`.
- Runs Optuna TPE for 200 trials, minimising mean squared error under 10-fold CV on the training split.
- Refits the best configuration on the full training split and evaluates once on the test split.
- Writes the `run_*.json` + Optuna `.log` in the structure shown above.

Because Optuna persists to disk via `JournalStorage`, an interrupted run picks
up exactly where it left off on the next invocation.

## Model keys (CLI ↔ paper table column)

| CLI `--model` | Paper column                          | Group         |
|---------------|---------------------------------------|---------------|
| `EBM`         | ExplainableBoostingRegressor          | Interpretable |
| `LGBM_interp` | LGBM                                  | Interpretable |
| `XGB_interp`  | XGB                                   | Interpretable |
| `SepALS_r2`   | SepALS (r ≤ 2)                        | Interpretable |
| `SepALS_r10`  | SepALS (r ≤ 10)                       | Interpretable |
| `TSL_1product`| TSL (1-product)                       | Interpretable |
| `TSL_R2`      | TSL (R ≤ 2)                           | Interpretable |
| `TSL_R10`     | TSL (R ≤ 10)                          | Interpretable |
| `LGBM_bb`     | LGBM                                  | Black-box     |
| `RF_bb`       | RandomForestRegressor                 | Black-box     |
| `XGB_bb`      | XGB                                   | Black-box     |

## Dataset task IDs

The 27 OpenML CTR 23 task IDs used in the paper benchmark are listed in
`scripts/tune_all.sh`. The full 35-suite metadata (name, n, p) is embedded in
`scripts/regenerate_summary.py` (`CTR23_DATASETS`).
