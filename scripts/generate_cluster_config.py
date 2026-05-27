#!/usr/bin/env python3
"""
Generate a flat JSON config enumerating every (model, dataset, seed) run for
cluster execution.

Each entry in ``runs`` is one atomic job: a single model tuned on a single
dataset under a single outer-split seed. A SLURM array job dispatches one task
per entry (``$SLURM_ARRAY_TASK_ID`` indexes directly into ``runs``), so the
whole benchmark — including repeated random splits — runs in parallel.

Seeds are derived from the count: ``--n-outer-trials N`` produces seeds
``range(N) = [0..N-1]`` for every (model, dataset) pair. Repeated splits let
``regenerate_summary.py`` report ``mean ± SE`` across seeds.

Usage:

    python scripts/generate_cluster_config.py \
        --models EBM XGB_bb \
        --task-ids 361617 361618 \
        --n-outer-trials 5 \
        --output cluster_config.json

With no ``--models`` / ``--task-ids`` it enumerates every runnable model and
every CTR 23 dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

# All three scripts live in scripts/, which is sys.path[0] when this file is run
# directly, so these sibling imports resolve without packaging.
from tune import CATEGORY_DIR
from regenerate_summary import CTR23_DATASETS


def build_runs(models, datasets, n_outer_trials):
    """Enumerate product(models, datasets, seeds) into a flat, sorted run list."""
    runs = []
    for ds, model, seed in product(datasets, models, range(n_outer_trials)):
        n, p = ds.get("n"), ds.get("p")
        n_times_p = n * p if (n and p) else None
        runs.append({
            "model": model,
            "task_id": ds["task_id"],
            "dataset_name": ds["name"],
            "seed": seed,
            "n_times_p": n_times_p,
        })
    # Smallest datasets first, then model, then seed — stable and deterministic.
    runs.sort(key=lambda r: (
        r["n_times_p"] is None,
        r["n_times_p"] or 0,
        r["model"],
        r["seed"],
    ))
    return runs


def main() -> None:
    here = Path(__file__).resolve().parent
    default_output = here.parent / "cluster_config.json"

    all_models = sorted(CATEGORY_DIR)
    all_task_ids = [ds["task_id"] for ds in CTR23_DATASETS]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=all_models, default=all_models,
                   help="Model keys to include (default: all runnable models).")
    p.add_argument("--task-ids", nargs="+", type=int, default=all_task_ids,
                   help="OpenML task IDs to include (default: all CTR 23 datasets).")
    p.add_argument("--n-outer-trials", type=int, default=1, metavar="N",
                   help="Number of repeated 80/20 splits per (model, dataset); "
                        "seeds are range(N) = [0..N-1]. Default 1.")
    p.add_argument("--n-trials", type=int, default=200,
                   help="Optuna HPO trials per run (inner search budget).")
    p.add_argument("--output", type=Path, default=default_output)
    args = p.parse_args()

    by_task_id = {ds["task_id"]: ds for ds in CTR23_DATASETS}
    unknown = [tid for tid in args.task_ids if tid not in by_task_id]
    if unknown:
        print(f"Unknown task IDs (not in CTR23_DATASETS): {unknown}", file=sys.stderr)
        sys.exit(1)
    datasets = [by_task_id[tid] for tid in args.task_ids]

    runs = build_runs(args.models, datasets, args.n_outer_trials)

    config = {
        "metadata": {
            "total_runs": len(runs),
            "n_models": len(args.models),
            "n_datasets": len(datasets),
            "n_outer_trials": args.n_outer_trials,
            "description": "Cluster config: one job per (model, dataset, seed).",
        },
        "global_config": {"n_trials": args.n_trials},
        "runs": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"  total_runs = {len(runs)} "
          f"({len(args.models)} models x {len(datasets)} datasets x "
          f"{args.n_outer_trials} seeds)", file=sys.stderr)


if __name__ == "__main__":
    main()
