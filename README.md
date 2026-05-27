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
    ├── tune.py                            tune one (dataset, model) cell (all seeds)
    ├── tune_all.sh                        local loop of tune.py over the 27 × 11 grid
    ├── generate_cluster_config.py         enumerate (model, dataset, seed) runs → config
    ├── run_cluster_experiment.py          run one (model, dataset, seed) from a config
    └── submit_cluster_jobs.sh             SLURM array submit + resume/requeue
```

## What's inside `results/`

Each `<dataset>/<ModelClass>/` directory contains, **per outer-split seed**:

- `run_seed<seed>_<dataset>_<ModelClass>.json` — best-trial summary for that
  seed's 80/20 split: `test_mse` / `test_rmse` on the held-out test split,
  `best_params`, `split_seed`, dataset metadata. With repeated splits there is
  one such file per seed (`run_seed0_…`, `run_seed1_…`, …).
- `<dataset>_<ModelClass>_seed<seed>.log` — Optuna JournalStorage log (JSONL)
  for that seed; contains every trial's sampled parameters, value, and state.
  Re-openable with `optuna.storages.JournalStorage(JournalFileBackend(path))`
  for full replay or trial-level analysis.

(Older single-split bundles may instead contain `run_<id>_…json` /
`<dataset>_<ModelClass>.log`; `regenerate_summary.py` reads both layouts.)

Excluded from the bundle (regeneratable from `best_params`):

- `*_best.bin` — pickled fitted models (~536 MB total)
- `*_best.sqlite` — TSL visualization databases (~52 GB total)
- The Rust core / Python bindings for TSL and SepALS — distributed separately.

## Reproduce the summary table

No external packages required for this step (Python ≥ 3.10 stdlib only):

```bash
python scripts/regenerate_summary.py
```

This walks `results/`, aggregates the test RMSE per `(dataset, model)`, and
writes `summary_interpretable_vs_blackbox.txt` — the table that the appendix's
full-benchmark figure is derived from.

When a cell has results from multiple outer-split seeds it is reported as
`mean ± SE`, where `SE = std(test_MSEs) / sqrt(n_seeds)` (standard error of the
mean MSE across seeds). A single-seed cell shows the bare RMSE.

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

Each invocation, for every outer-split seed `0 … N-1` (where `N` =
`--n-outer-trials`, default 1):
- Loads the OpenML task, applies an 80/20 train/test split with `random_state=seed`.
- Builds the paper-aligned search space inline in `tune.py`.
- Runs Optuna TPE for 200 trials, minimising mean squared error under 10-fold CV on the training split.
- Refits the best configuration on the full training split and evaluates once on the test split.
- Writes `run_seed<seed>_*.json` + the per-seed Optuna `.log` in the structure shown above.

Because Optuna persists to disk via `JournalStorage`, an interrupted run picks
up exactly where it left off on the next invocation — per seed. Result files are
never deleted, so finished seeds are kept and only outstanding work resumes.

### Repeated random splits (recommended)

A single 80/20 split is a noisy estimate. To repeat the outer split and obtain
`mean ± SE`, set the number of outer trials (seeds become `0 … N-1`):

```bash
N_OUTER_TRIALS=5 bash scripts/tune_all.sh            # local, sequential
```

### Cluster execution (massively parallel)

Each `(model, dataset, seed)` triple is an independent job, dispatched as one
task of a SLURM **array job**. Three steps:

```bash
# 1. Enumerate every run into a flat config (one entry per model × dataset × seed).
python scripts/generate_cluster_config.py --n-outer-trials 5 --output cluster_config.json

# 2. Submit the array; $SLURM_ARRAY_TASK_ID indexes directly into config["runs"].
#    Resources are env-overridable; PARTITION/ACCOUNT emit their #SBATCH line only if set.
#    MAX_CONCURRENT caps simultaneously running tasks (appends %N to the array spec).
CPUS=8 MEM=32G TIME=24:00:00 PARTITION=batch MAX_CONCURRENT=64 \
  bash scripts/submit_cluster_jobs.sh cluster_config.json results

# 3. Rebuild the table (mean ± SE across seeds).
python scripts/regenerate_summary.py --results-dir results
```

Jobs are submitted with `--requeue` and `--open-mode=append`, so a preempted
task is automatically requeued and resumes from its own per-seed Optuna `.log`
(only an in-flight trial is redone). For very large sweeps, mind your cluster's
`MaxArraySize` (`scontrol show config | grep MaxArraySize`) and submit in
index-range chunks via the optional 3rd argument, e.g.
`bash scripts/submit_cluster_jobs.sh cluster_config.json results 0-999`.

Re-running step 2 against an existing `results/` directory **resumes**: it
resubmits only runs without a result JSON (timed-out / OOM-killed / failed /
never-started — the SLURM logs are parsed to report which), and each requeued
job continues from its own per-seed Optuna `.log`. The submit script is generic
SLURM (no site-specific container, partition, or environment baked in); set
`PYTHON`, `CPUS`, `MEM`, `TIME`, `PARTITION`, `ACCOUNT` as needed.

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
