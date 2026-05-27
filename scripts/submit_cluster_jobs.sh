#!/bin/bash
# Submit one SLURM array task per (model, dataset, seed) run in a cluster config.
#
# Usage:
#   bash scripts/submit_cluster_jobs.sh <config.json> <results_dir> [run_ids]
#     - run_ids (optional): comma-separated 0-indexed run IDs to submit (e.g. "0" or "0,3,7")
#
# Resource / environment knobs (env vars, neutral defaults; nothing here is
# tied to a specific cluster):
#   CPUS=4  MEM=16G  TIME=24:00:00  PARTITION=  ACCOUNT=  PYTHON=python
# PARTITION / ACCOUNT emit their #SBATCH line only when set, so the script is
# portable across SLURM sites.
#
# Resume/requeue (when the results dir already exists): resubmits every run that
# has no result JSON yet — covering timed-out (DUE TO TIME LIMIT), OOM-killed
# (oom_kill / OOM Killed), failed (run_*_ERROR.json), and never-started runs;
# the SLURM logs are parsed only to report why. Each requeued job resumes its own
# per-seed Optuna .log; failed runs get that log removed so HPO restarts clean.
# Completed runs (result JSON present) are skipped.

set -u

CONFIG_FILE=$1
OUTPUT_DIR=$2
RUN_IDS=${3:-}

CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
TIME="${TIME:-}"                      # unset -> no --time line (use the partition default)
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
PYTHON="${PYTHON:-python}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap simultaneously running array tasks (appends %N)

CURRENT_DIR=$(pwd)
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$CURRENT_DIR/$1" ;; esac; }
ABS_CONFIG=$(abspath "$CONFIG_FILE")
ABS_OUTPUT=$(abspath "$OUTPUT_DIR")

# Optional #SBATCH directives, emitted only when the var is non-empty.
OPT_DIRECTIVES=""
[ -n "$PARTITION" ] && OPT_DIRECTIVES+="#SBATCH --partition=$PARTITION"$'\n'
[ -n "$ACCOUNT" ]   && OPT_DIRECTIVES+="#SBATCH --account=$ACCOUNT"$'\n'
[ -n "$TIME" ]      && OPT_DIRECTIVES+="#SBATCH --time=$TIME"$'\n'

TOTAL_RUNS=$($PYTHON -c "import json; print(json.load(open('$ABS_CONFIG'))['metadata']['total_runs'])")
ARRAY_LIMIT=$((TOTAL_RUNS - 1))

# Collapse a sorted, unique list of ints (stdin, one per line) into a compact
# SLURM array spec with ranges, e.g. "0 1 2 5" -> "0-2,5". A bare comma list of
# thousands of ids overflows sbatch's spec-length limit; ranges stay short.
compress_ranges() {
    awk '
        NR==1 { s=$1; p=$1; next }
        $1==p+1 { p=$1; next }
        { printf "%s%s", (c++?",":""), (s==p?s:s"-"p); s=$1; p=$1 }
        END { if (NR) printf "%s%s", (c++?",":""), (s==p?s:s"-"p) }
    '
}

# Emit a SLURM array submission for the given array spec (e.g. "0-41" or "3,7,9").
submit_array() {
    local array_spec=$1
    [ -n "$MAX_CONCURRENT" ] && array_spec="${array_spec}%${MAX_CONCURRENT}"
    mkdir -p "$ABS_OUTPUT/logs"
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tsl_bench
#SBATCH --array=$array_spec
#SBATCH --cpus-per-task=$CPUS
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=$MEM
#SBATCH --requeue
#SBATCH --open-mode=append
${OPT_DIRECTIVES}#SBATCH --output=$ABS_OUTPUT/logs/job_%A_%a.out
#SBATCH --error=$ABS_OUTPUT/logs/job_%A_%a.err

echo "Task ID: \$SLURM_ARRAY_TASK_ID on node \$(hostname)"

$PYTHON scripts/run_cluster_experiment.py \\
    --config "$ABS_CONFIG" \\
    --run-id \$SLURM_ARRAY_TASK_ID \\
    --results-root "$ABS_OUTPUT"
EOT
}

# Print, one per line, the run-ids that still need running and (separately) those
# that failed. Completeness is judged by the presence of the per-(model,dataset,
# seed) result JSON in the standard results layout — robust to partial runs.
# Output format:  "MISSING <id>" or "FAILED <id>" lines.
scan_runs() {
    $PYTHON - "$ABS_CONFIG" "$ABS_OUTPUT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from tune import CATEGORY_DIR

cfg_path, out_root = sys.argv[1], Path(sys.argv[2])
runs = json.load(open(cfg_path))["runs"]
for rid, r in enumerate(runs):
    cat, var, mc = CATEGORY_DIR[r["model"]]
    ds = r["dataset_name"]
    leaf = out_root / cat / var / ds / mc
    result = leaf / f"run_seed{r['seed']}_{ds}_{mc}.json"
    if result.exists():
        continue  # done
    err = leaf / f"run_{rid:04d}_ERROR.json"
    print(f"{'FAILED' if err.exists() else 'MISSING'} {rid}")
PY
}

# Print the per-seed Optuna log path (relative to the results root) for a run id.
log_path_for_run() {
    local run_id=$1
    $PYTHON - "$ABS_CONFIG" "$run_id" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from tune import CATEGORY_DIR
cfg, run_id = sys.argv[1], int(sys.argv[2])
r = json.load(open(cfg))["runs"][run_id]
cat, var, mc = CATEGORY_DIR[r["model"]]
ds = r["dataset_name"]
print(f"{cat}/{var}/{ds}/{mc}/{ds}_{mc}_seed{r['seed']}.log")
PY
}

# Report (for visibility only) timed-out / OOM run-ids by parsing SLURM logs.
report_log_reasons() {
    local logs_dir="$ABS_OUTPUT/logs"
    [ -d "$logs_dir" ] || return 0
    shopt -s nullglob

    # Timed-out: a .err with "DUE TO TIME LIMIT"; map to run-id via its .out header.
    local out_file err_file run_id
    for err_file in "$logs_dir"/*.err; do
        grep -q "DUE TO TIME LIMIT" "$err_file" 2>/dev/null || continue
        out_file="${err_file%.err}.out"
        [ -f "$out_file" ] || continue
        run_id=$(grep -m 1 "^Run ID:" "$out_file" 2>/dev/null | sed -n 's/^Run ID: \([0-9]*\)\/.*/\1/p')
        [ -n "$run_id" ] && echo "  timed-out: run $((run_id - 1))"
    done

    # OOM: stderr files job_<jobid>_<task>.err; report the latest job id per task.
    # (Avoids bash-4 associative arrays for portability.)
    local oom_lines=""
    local base jid tid
    for err_file in "$logs_dir"/job_*.err; do
        base=$(basename "$err_file")
        [[ "$base" =~ ^job_([0-9]+)_([0-9]+)\.err$ ]] || continue
        jid="${BASH_REMATCH[1]}"; tid="${BASH_REMATCH[2]}"
        { [ "$tid" -lt 0 ] || [ "$tid" -ge "$TOTAL_RUNS" ]; } && continue
        oom_lines+="$tid $jid $err_file"$'\n'
    done
    # Keep only the highest job id per task, then check that file for OOM.
    printf '%s' "$oom_lines" | sort -k1,1n -k2,2n | awk '{a[$1]=$3} END{for (t in a) print t, a[t]}' \
    | while read -r tid latest_err; do
        [ -z "$tid" ] && continue
        if grep -qiE 'oom_kill|OOM Killed' "$latest_err" 2>/dev/null; then
            echo "  OOM-killed: run $tid"
        fi
    done

    shopt -u nullglob
}

# --- Explicit run-id list: submit exactly those and exit ---------------------
if [ -n "$RUN_IDS" ]; then
    echo "Submitting selected run IDs only (Array: $RUN_IDS)..."
    mkdir -p "$ABS_OUTPUT"
    cp -n "$ABS_CONFIG" "$ABS_OUTPUT/$(basename "$ABS_CONFIG")" 2>/dev/null || true
    submit_array "$RUN_IDS"
    echo "Submitted run IDs: $RUN_IDS"
    exit 0
fi

# --- Resume/requeue when the results dir already exists ----------------------
if [ -d "$ABS_OUTPUT" ]; then
    echo "Results dir exists. Scanning for incomplete / failed runs..."
    report_log_reasons

    requeue=()
    n_failed=0
    while read -r kind rid; do
        [ -z "$rid" ] && continue
        requeue+=("$rid")
        if [ "$kind" = "FAILED" ]; then
            n_failed=$((n_failed + 1))
            echo "  failed: run $rid (removing its Optuna log so HPO restarts clean)"
            rel_log=$(log_path_for_run "$rid")
            [ -n "$rel_log" ] && [ -f "$ABS_OUTPUT/$rel_log" ] && rm -f "$ABS_OUTPUT/$rel_log"
            # Clear the stale ERROR marker so a clean run can be detected next time.
            leaf_dir=$(dirname "$ABS_OUTPUT/$rel_log")
            rm -f "$leaf_dir/run_$(printf '%04d' "$rid")_ERROR.json"
        fi
    done < <(scan_runs)

    if [ "${#requeue[@]}" -gt 0 ]; then
        # Dedup + sort.
        IFS=$'\n' requeue=($(printf '%s\n' "${requeue[@]}" | sort -n | uniq)); unset IFS
        echo "Requeuing ${#requeue[@]} runs (${n_failed} failed, rest missing/incomplete)."
        array_spec=$(printf '%s\n' "${requeue[@]}" | compress_ranges)
        cp -n "$ABS_CONFIG" "$ABS_OUTPUT/$(basename "$ABS_CONFIG")" 2>/dev/null || true
        submit_array "$array_spec"
        echo "Resubmitted: $array_spec"
        exit 0
    fi

    echo "All runs complete. Nothing to requeue."
    exit 0
fi

# --- First submission: the whole array ---------------------------------------
echo "Submitting $TOTAL_RUNS jobs (Array 0-$ARRAY_LIMIT)..."
mkdir -p "$ABS_OUTPUT"
cp "$ABS_CONFIG" "$ABS_OUTPUT/$(basename "$ABS_CONFIG")"
submit_array "0-$ARRAY_LIMIT"

# Example end-to-end (run from the repo root):
#   python scripts/generate_cluster_config.py --n-outer-trials 5 --output cluster_config.json
#   CPUS=8 MEM=32G PARTITION=batch bash scripts/submit_cluster_jobs.sh cluster_config.json results
#   python scripts/regenerate_summary.py --results-dir results
