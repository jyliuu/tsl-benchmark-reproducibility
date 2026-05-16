#!/usr/bin/env python3
"""
Regenerate the interpretable-vs-blackbox summary table directly from the raw
``run_*.json`` result files shipped under ``results/``.

This is the standalone version of the analogous script in the hyperparameter
tuning dashboard. It does not depend on the dashboard SQLite database, Flask,
SQLAlchemy, or any private packages; only the standard library plus optional
``numpy``/``pandas`` are needed (neither is used here).

Usage (from the project root):

    python scripts/regenerate_summary.py \
        --results-dir results \
        --output summary_interpretable_vs_blackbox.txt

The output format matches the table in the paper appendix's full benchmark
results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------------
# CTR 23 suite metadata: (task_id, dataset_name, n, p)
#
# Extracted once from OpenML suite_id 353. ``p`` is the number of input
# features (OpenML's reported "features" count minus the target column),
# matching the convention used throughout the paper.
# -------------------------------------------------------------------------

CTR23_DATASETS: List[Dict] = [
    {"task_id": 361234, "name": "abalone",                       "n":  4177, "p":  8},
    {"task_id": 361235, "name": "airfoil_self_noise",            "n":  1503, "p":  5},
    {"task_id": 361236, "name": "auction_verification",          "n":  2043, "p":  7},
    {"task_id": 361237, "name": "concrete_compressive_strength", "n":  1030, "p":  8},
    {"task_id": 361241, "name": "physiochemical_protein",        "n": 45730, "p":  9},
    {"task_id": 361242, "name": "superconductivity",             "n": 21263, "p": 81},
    {"task_id": 361243, "name": "geographical_origin_of_music",  "n":  1059, "p": 116},
    {"task_id": 361244, "name": "solar_flare",                   "n":  1066, "p": 10},
    {"task_id": 361247, "name": "naval_propulsion_plant",        "n": 11934, "p": 14},
    {"task_id": 361249, "name": "white_wine",                    "n":  4898, "p": 11},
    {"task_id": 361250, "name": "red_wine",                      "n":  1599, "p": 11},
    {"task_id": 361251, "name": "grid_stability",                "n": 10000, "p": 12},
    {"task_id": 361252, "name": "video_transcoding",             "n": 68784, "p": 18},
    {"task_id": 361253, "name": "wave_energy",                   "n": 72000, "p": 48},
    {"task_id": 361254, "name": "sarcos",                        "n": 48933, "p": 21},
    {"task_id": 361255, "name": "california_housing",            "n": 20640, "p":  8},
    {"task_id": 361256, "name": "cpu_activity",                  "n":  8192, "p": 21},
    {"task_id": 361257, "name": "diamonds",                      "n": 53940, "p":  9},
    {"task_id": 361258, "name": "kin8nm",                        "n":  8192, "p":  8},
    {"task_id": 361259, "name": "pumadyn32nh",                   "n":  8192, "p": 32},
    {"task_id": 361260, "name": "miami_housing",                 "n": 13932, "p": 15},
    {"task_id": 361261, "name": "cps88wages",                    "n": 28155, "p":  6},
    {"task_id": 361264, "name": "socmob",                        "n":  1156, "p":  5},
    {"task_id": 361266, "name": "kings_county",                  "n": 21613, "p": 21},
    {"task_id": 361267, "name": "brazilian_houses",              "n": 10692, "p":  9},
    {"task_id": 361268, "name": "fps_benchmark",                 "n": 24624, "p": 43},
    {"task_id": 361269, "name": "health_insurance",              "n": 22272, "p": 11},
    {"task_id": 361272, "name": "fifa",                          "n": 19178, "p": 28},
    {"task_id": 361616, "name": "Moneyball",                     "n":  1232, "p": 14},
    {"task_id": 361617, "name": "energy_efficiency",             "n":   768, "p":  8},
    {"task_id": 361618, "name": "forest_fires",                  "n":   517, "p": 12},
    {"task_id": 361619, "name": "student_performance_por",       "n":   649, "p": 30},
    {"task_id": 361621, "name": "QSAR_fish_toxicity",            "n":   908, "p":  6},
    {"task_id": 361622, "name": "cars",                          "n":   804, "p": 17},
    {"task_id": 361623, "name": "space_ga",                      "n":  3107, "p":  6},
]

# -------------------------------------------------------------------------
# Directory layout → (column display name, group)
#
# Each top-level subdir under ``results/`` represents one model variant.
# Both TSL variants (R ≤ 2 and R ≤ 10) are reported under the Interpretable
# group.
# -------------------------------------------------------------------------

PATH_TO_DISPLAY = {
    ("interpretable", "tsl_r2"):        ("TSL (R ≤ 2)",    "interpretable"),
    ("interpretable", "tsl_1product"):  ("TSL (1-product)", "interpretable"),
    ("interpretable", "sepals_r2"):     ("Sepals (r ≤ 2)",  "interpretable"),
    ("interpretable", "sepals_r10"):    ("Sepals (r ≤ 10)", "interpretable"),
    # interpretable/others houses EBM, LGBM (interp), XGB (interp); branch on model
    ("interpretable", "others"):        None,
    ("interpretable", "tsl_r10"):       ("TSL (R ≤ 10)",   "interpretable"),
    # blackbox/others houses LGBM (bb), RF (bb), XGB (bb)
    ("blackbox",      "others"):        None,
}

OTHERS_INTERPRETABLE = {
    "ExplainableBoostingRegressor": ("ExplainableBoostingRegressor", "interpretable"),
    "LGBMRegressor":                ("LGBMRegressor",                "interpretable"),
    "XGBRegressor":                 ("XGBRegressor",                 "interpretable"),
}

OTHERS_BLACKBOX = {
    "LGBMRegressor":         ("LGBMRegressor",         "blackbox"),
    "RandomForestRegressor": ("RandomForestRegressor", "blackbox"),
    "XGBRegressor":          ("XGBRegressor",          "blackbox"),
}

# Column orders within each group, matching the paper's full benchmark table.
INTERP_COLUMN_ORDER = [
    "ExplainableBoostingRegressor",
    "LGBMRegressor",
    "XGBRegressor",
    "Sepals (r ≤ 2)",
    "Sepals (r ≤ 10)",
    "TSL (1-product)",
    "TSL (R ≤ 2)",
    "TSL (R ≤ 10)",
]
BLACKBOX_COLUMN_ORDER = ["LGBMRegressor", "RandomForestRegressor", "XGBRegressor"]


# -------------------------------------------------------------------------
# Result loading
# -------------------------------------------------------------------------

def _row_for_run(
    path: Path,
    record: dict,
    results_dir: Path,
) -> Optional[Tuple[int, str, str, float]]:
    """Return (task_id, group, column_label, test_rmse) for one run JSON.

    Returns None if the run can't be mapped (e.g. unknown directory layout,
    missing fields, or no successful test_rmse).
    """
    try:
        rel = path.relative_to(results_dir).parts
    except ValueError:
        return None
    if len(rel) < 4:
        return None
    group_dir, variant_dir = rel[0], rel[1]
    model_class = record.get("model") or rel[-2]

    mapping = PATH_TO_DISPLAY.get((group_dir, variant_dir))
    if mapping is None:
        if (group_dir, variant_dir) == ("interpretable", "others"):
            mapping = OTHERS_INTERPRETABLE.get(model_class)
        elif (group_dir, variant_dir) == ("blackbox", "others"):
            mapping = OTHERS_BLACKBOX.get(model_class)
    if mapping is None:
        return None
    column_label, group = mapping

    task_id = (record.get("dataset_config") or {}).get("task_id")
    if task_id is None:
        return None

    # Prefer mean across folds when available; fall back to single-fold test_rmse.
    if record.get("mean_test_rmse") is not None:
        test_rmse = record["mean_test_rmse"]
    elif record.get("test_rmse") is not None:
        test_rmse = record["test_rmse"]
    else:
        results = record.get("results") or []
        rmses = [r.get("test_rmse") for r in results if r.get("test_rmse") is not None]
        if not rmses:
            return None
        test_rmse = sum(rmses) / len(rmses)

    return (task_id, group, column_label, float(test_rmse))


def load_results(results_dir: Path) -> Dict[Tuple[int, str, str], List[float]]:
    """Walk ``results_dir`` and collect a {(task_id, group, column_label): [rmse, ...]} dict."""
    bucket: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    results_dir = results_dir.resolve()
    for run_path in results_dir.rglob("run_*.json"):
        if run_path.name.endswith("_ERROR.json"):
            continue
        try:
            with open(run_path) as f:
                record = json.load(f)
        except Exception as e:
            print(f"WARNING: failed to parse {run_path}: {e}", file=sys.stderr)
            continue
        row = _row_for_run(run_path.resolve(), record, results_dir)
        if row is None:
            continue
        task_id, group, column_label, test_rmse = row
        bucket[(task_id, group, column_label)].append(test_rmse)
    return bucket


# -------------------------------------------------------------------------
# Table rendering (matches the dashboard's summary output line-for-line)
# -------------------------------------------------------------------------

def _format_dataset_display(name: str, n: Optional[int], p: Optional[int], width: int) -> str:
    n_str = str(n) if n is not None else "N/A"
    p_str = str(p) if p is not None else "N/A"
    s = f"{name} (n: {n_str}, p: {p_str})"
    return s[: width - 3] + "..." if len(s) > width else s


def render_summary(
    bucket: Dict[Tuple[int, str, str], List[float]],
    out_stream,
) -> None:
    # Per-(dataset, column) mean across runs.
    rmse_by_ds: Dict[int, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for (task_id, group, column_label), values in bucket.items():
        rmse_by_ds[task_id][(group, column_label)] = sum(values) / len(values)

    # Sort datasets by n×p (paper's display order) with unknowns to the end.
    rows = []
    for ds in CTR23_DATASETS:
        np_prod = ds["n"] * ds["p"] if ds["n"] and ds["p"] else None
        rows.append({**ds, "n_times_p": np_prod})
    rows.sort(key=lambda r: (r["n_times_p"] is None, r["n_times_p"] or 0))

    # Column widths.
    model_col_width = max(
        16,
        max(len(c) for c in INTERP_COLUMN_ORDER + BLACKBOX_COLUMN_ORDER),
    )
    dataset_col_width = 50
    interp_width  = len(INTERP_COLUMN_ORDER)  * (model_col_width + 1) - 1
    blackb_width  = len(BLACKBOX_COLUMN_ORDER) * (model_col_width + 1) - 1
    total_width   = dataset_col_width + 3 + interp_width + 3 + blackb_width

    def line(s=""):
        print(s, file=out_stream)

    line("=" * total_width)
    line("SUMMARY: Interpretable vs Blackbox (by Model) - CTR 23 Suite Datasets (suite_id 353)")
    line("=" * total_width)
    line()

    header1 = f"{'Dataset':<{dataset_col_width}} | "
    header1 += f"{'Interpretable':<{interp_width}} | "
    header1 += f"{'Blackbox':<{blackb_width}}"
    line(header1)

    header2 = "-" * dataset_col_width + " | "
    header2 += " | ".join(f"{c:<{model_col_width}}" for c in INTERP_COLUMN_ORDER)
    header2 += " | "
    header2 += " | ".join(f"{c:<{model_col_width}}" for c in BLACKBOX_COLUMN_ORDER)
    line(header2)
    line("-" * total_width)

    # Ranking counters (within group + overall) — counted only on rows up through
    # kings_county, matching the paper's "27 evaluated datasets" full table.
    kings_cutoff_idx: Optional[int] = None
    for i, r in enumerate(rows):
        if r["name"] == "kings_county" and r["n"] == 21613:
            kings_cutoff_idx = i
            break

    interp_rank = defaultdict(lambda: {"best": 0, "second": 0, "third": 0})
    bb_rank     = defaultdict(lambda: {"best": 0, "second": 0, "third": 0})
    overall_rank= defaultdict(lambda: {"best": 0, "second": 0, "third": 0})
    interp_avg_rank = defaultdict(lambda: {"sum": 0.0, "n": 0})
    bb_avg_rank     = defaultdict(lambda: {"sum": 0.0, "n": 0})

    for ds_idx, ds in enumerate(rows):
        task_id = ds["task_id"]
        ds_rmses = rmse_by_ds.get(task_id, {})
        interp_vals = {c: ds_rmses.get(("interpretable", c)) for c in INTERP_COLUMN_ORDER}
        bb_vals     = {c: ds_rmses.get(("blackbox", c))     for c in BLACKBOX_COLUMN_ORDER}

        all_present = [v for v in list(interp_vals.values()) + list(bb_vals.values()) if v is not None]
        sorted_unique = sorted(set(all_present))
        top3 = sorted_unique[:3]
        rmse_markers = {}
        if len(top3) > 0: rmse_markers[top3[0]] = "(***)"
        if len(top3) > 1: rmse_markers[top3[1]] = "(**)"
        if len(top3) > 2: rmse_markers[top3[2]] = "(*)"

        within_window = kings_cutoff_idx is None or ds_idx <= kings_cutoff_idx

        # Interpretable-only and blackbox-only within-group rankings (for the
        # group leaderboards at the bottom).
        def _group_marks_and_ranks(group_vals):
            present = [(name, v) for name, v in group_vals.items() if v is not None]
            if not present:
                return {}, {}
            uniq_sorted = sorted({v for _, v in present})
            marks = {}
            if len(uniq_sorted) > 0: marks[uniq_sorted[0]] = "best"
            if len(uniq_sorted) > 1: marks[uniq_sorted[1]] = "second"
            if len(uniq_sorted) > 2: marks[uniq_sorted[2]] = "third"
            rank_idx = {v: i + 1 for i, v in enumerate(uniq_sorted)}
            return marks, rank_idx

        interp_marks, interp_rmse_to_rank = _group_marks_and_ranks(interp_vals)
        bb_marks,     bb_rmse_to_rank     = _group_marks_and_ranks(bb_vals)

        # Build the printable row
        ds_display = _format_dataset_display(ds["name"], ds["n"], ds["p"], dataset_col_width)
        cells_interp, cells_bb = [], []
        for col in INTERP_COLUMN_ORDER:
            v = interp_vals.get(col)
            if v is None:
                cells_interp.append("✗")
            else:
                marker = rmse_markers.get(v, "")
                cells_interp.append(f"{v:.4f} {marker}".strip())
                if within_window:
                    g = interp_marks.get(v)
                    if g == "best":   interp_rank[col]["best"]   += 1
                    elif g == "second": interp_rank[col]["second"] += 1
                    elif g == "third":  interp_rank[col]["third"]  += 1
                    if marker == "(***)": overall_rank[col]["best"] += 1
                    elif marker == "(**)": overall_rank[col]["second"] += 1
                    elif marker == "(*)":  overall_rank[col]["third"]  += 1
                    rk = interp_rmse_to_rank.get(v)
                    if rk is not None:
                        interp_avg_rank[col]["sum"] += rk
                        interp_avg_rank[col]["n"]   += 1

        for col in BLACKBOX_COLUMN_ORDER:
            v = bb_vals.get(col)
            if v is None:
                cells_bb.append("✗")
            else:
                marker = rmse_markers.get(v, "")
                cells_bb.append(f"{v:.4f} {marker}".strip())
                if within_window:
                    g = bb_marks.get(v)
                    if g == "best":   bb_rank[col]["best"]   += 1
                    elif g == "second": bb_rank[col]["second"] += 1
                    elif g == "third":  bb_rank[col]["third"]  += 1
                    if marker == "(***)": overall_rank[col]["best"] += 1
                    elif marker == "(**)": overall_rank[col]["second"] += 1
                    elif marker == "(*)":  overall_rank[col]["third"]  += 1
                    rk = bb_rmse_to_rank.get(v)
                    if rk is not None:
                        bb_avg_rank[col]["sum"] += rk
                        bb_avg_rank[col]["n"]   += 1

        row_str = f"{ds_display:<{dataset_col_width}} | "
        row_str += " | ".join(f"{c:>{model_col_width}}" for c in cells_interp)
        row_str += " | "
        row_str += " | ".join(f"{c:>{model_col_width}}" for c in cells_bb)
        line(row_str)

    line("-" * total_width)

    # Per-model coverage summary.
    total = len(rows)
    line()
    line("Summary:")
    line(f"  Total datasets in CTR 23 suite: {total}")
    line()
    line("  Interpretable models:")
    for c in INTERP_COLUMN_ORDER:
        n_done = sum(1 for ds in rows if rmse_by_ds.get(ds["task_id"], {}).get(("interpretable", c)) is not None)
        line(f"    {c}: {n_done} completed, {total - n_done} missing")
    line()
    line("  Blackbox models:")
    for c in BLACKBOX_COLUMN_ORDER:
        n_done = sum(1 for ds in rows if rmse_by_ds.get(ds["task_id"], {}).get(("blackbox", c)) is not None)
        line(f"    {c}: {n_done} completed, {total - n_done} missing")

    # Ranking leaderboards.
    line()
    line("=" * total_width)
    line("RANKING SUMMARY")
    line("=" * total_width)

    def _print_group(title, columns, rank_dict, avg_rank_dict):
        line(); line(f"{title}:")
        ordering = sorted(
            columns,
            key=lambda c: (
                -rank_dict[c]["best"],
                -rank_dict[c]["second"],
                -rank_dict[c]["third"],
            ),
        )
        line(f"{'Model':<30} | {'Best (***)':>12} | {'Second (**)':>12} | {'Third (*)':>12} | {'Avg Rank':>10}")
        line("-" * 85)
        for c in ordering:
            r = rank_dict[c]
            a = avg_rank_dict[c]
            avg_str = f"{a['sum'] / a['n']:.2f}" if a["n"] else "N/A"
            line(f"{c:<30} | {r['best']:>12} | {r['second']:>12} | {r['third']:>12} | {avg_str:>10}")

    _print_group("Interpretable Group Rankings", INTERP_COLUMN_ORDER, interp_rank, interp_avg_rank)
    _print_group("Blackbox Group Rankings",      BLACKBOX_COLUMN_ORDER, bb_rank,    bb_avg_rank)

    line(); line("Overall Rankings (across all models):")
    all_cols = sorted(set(INTERP_COLUMN_ORDER + BLACKBOX_COLUMN_ORDER))
    ordering = sorted(
        all_cols,
        key=lambda c: (
            -overall_rank[c]["best"],
            -overall_rank[c]["second"],
            -overall_rank[c]["third"],
        ),
    )
    line(f"{'Model':<30} | {'Best (***)':>12} | {'Second (**)':>12} | {'Third (*)':>12}")
    line("-" * 70)
    for c in ordering:
        r = overall_rank[c]
        line(f"{c:<30} | {r['best']:>12} | {r['second']:>12} | {r['third']:>12}")
    line("=" * total_width)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_results = here.parent / "results"
    default_output = here.parent / "summary_interpretable_vs_blackbox.txt"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=default_results,
                   help=f"Root of the results tree (default: {default_results})")
    p.add_argument("--output", type=Path, default=default_output,
                   help=f"Where to write the table (default: {default_output}; '-' for stdout)")
    args = p.parse_args()

    if not args.results_dir.is_dir():
        print(f"results-dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    bucket = load_results(args.results_dir)
    print(f"Loaded {sum(len(v) for v in bucket.values())} run records across "
          f"{len({k[0] for k in bucket})} datasets and "
          f"{len({(k[1], k[2]) for k in bucket})} model variants.", file=sys.stderr)

    if str(args.output) == "-":
        render_summary(bucket, sys.stdout)
    else:
        with open(args.output, "w") as f:
            render_summary(bucket, f)
        print(f"Wrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
