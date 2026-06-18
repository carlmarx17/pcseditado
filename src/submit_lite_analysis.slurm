#!/bin/bash -l
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lite_an
#SBATCH --output=/cosma7/data/dp433/dc-mart18/lite_runs/lite_analysis_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/lite_runs/lite_analysis_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
ANALYSIS_SRC="$REPO/CodeforAnalisys/lite_instability_analysis.py"

module purge
module load gnu_comp/14.1.0

export OMP_NUM_THREADS=1

run_latest_case() {
  local case_name="$1"
  local run_dir
  run_dir="$(find "$BASE/lite_runs" -maxdepth 1 -mindepth 1 -type d -name "${case_name}_*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$run_dir" ]]; then
    echo "No run directory found for $case_name"
    return 0
  fi

  local outdir="$run_dir/analysis_job_${SLURM_JOB_ID}"
  mkdir -p "$outdir"
  echo "case=$case_name"
  echo "run_dir=$run_dir"
  PSC_PROFILE="$case_name" python "$ANALYSIS_SRC" --data-dir "$run_dir" --outdir "$outdir"
  echo "latest_rows for $case_name"
  tail -n 3 "$outdir/lite_instability_evolution.csv"
}

run_latest_case M_lite
run_latest_case F_lite
run_latest_case W_lite
