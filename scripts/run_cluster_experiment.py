#!/usr/bin/env python3
"""
Run a single (model, dataset, seed) job from a cluster config.

Resolves ``config["runs"][run_id]`` and executes exactly that one outer trial
via ``tune.run_seed``. Designed to be the body of a SLURM array task:

    python scripts/run_cluster_experiment.py \
        --config cluster_config.json \
        --run-id $SLURM_ARRAY_TASK_ID \
        --results-root results

The result ``run_seed{seed}_{dataset}_{model}.json`` lands in the standard
``results/{category}/{variant}/{dataset}/{ModelClass}/`` layout (computed inside
``run_seed``), so ``regenerate_summary.py`` picks it up unchanged. On failure a
``run_{run_id:04d}_ERROR.json`` is written so the submit script can requeue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import traceback
from pathlib import Path

# scripts/ is sys.path[0] when run directly, so this sibling import resolves.
from tune import CATEGORY_DIR, load_task, run_seed


def main() -> None:
    here = Path(__file__).resolve().parent
    results_default = here.parent / "results"

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to the cluster config JSON (see generate_cluster_config.py).")
    p.add_argument("--run-id", type=int, required=True,
                   help="0-indexed position into config['runs'].")
    p.add_argument("--results-root", type=Path, default=results_default)
    args = p.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)
    runs = config.get("runs", [])
    total = len(runs)

    if not (0 <= args.run_id < total):
        print(f"Invalid run-id {args.run_id}; must be in [0, {total - 1}].", file=sys.stderr)
        sys.exit(1)

    r = runs[args.run_id]
    n_trials = config.get("global_config", {}).get("n_trials", 200)

    args.results_root.mkdir(parents=True, exist_ok=True)

    # Header lines parsed by submit_cluster_jobs.sh's requeue scanner — keep the
    # "Configuration:" and "Run ID:" prefixes stable.
    print("=" * 80)
    print(f"Configuration: {args.config.resolve()}")
    print(f"Run ID: {args.run_id + 1}/{total}")
    print(f"Run: model={r['model']} task_id={r['task_id']} "
          f"dataset={r['dataset_name']} seed={r['seed']}")
    print("=" * 80, flush=True)

    # Copy the config into the results root once, for record-keeping.
    config_copy = args.results_root / args.config.name
    if not config_copy.exists():
        try:
            shutil.copy2(args.config, config_copy)
        except Exception as e:  # noqa: BLE001 - best-effort record-keeping
            print(f"Warning: could not copy config: {e}", file=sys.stderr)

    try:
        dataset_name, X_proc, y_arr = load_task(r["task_id"])
        run_seed(r["model"], r["task_id"], r["seed"], args.results_root,
                 dataset_name, X_proc, y_arr, n_trials=n_trials)
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED run-id {args.run_id}:\n{error_msg}", file=sys.stderr)
        # Place the ERROR file next to where the result would have gone, so it is
        # discoverable, falling back to the results root if the layout is unknown.
        try:
            category, variant, model_class_name = CATEGORY_DIR[r["model"]]
            err_dir = (args.results_root / category / variant
                       / r["dataset_name"] / model_class_name)
        except Exception:
            err_dir = args.results_root
        err_dir.mkdir(parents=True, exist_ok=True)
        err_path = err_dir / f"run_{args.run_id:04d}_ERROR.json"
        with open(err_path, "w") as f:
            json.dump({
                "success": False,
                "run_id": args.run_id,
                "run": r,
                "error": error_msg,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }, f, indent=2)
        print(f"Wrote {err_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
