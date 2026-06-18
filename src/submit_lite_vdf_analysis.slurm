#!/bin/bash -l
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=lite_vdf
#SBATCH --output=/cosma7/data/dp433/dc-mart18/lite_runs/lite_vdf_analysis_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/lite_runs/lite_vdf_analysis_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
ANALYSIS_DIR="$REPO/CodeforAnalisys"
RESULTS_ROOT="$BASE/lite_runs/lite_vdf_analysis_${SLURM_JOB_ID}"

module purge
module load gnu_comp/14.1.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/psc-matplotlib-vdf-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_ROOT"

run_case() {
  local case_name="$1"
  local run_dir
  run_dir="$(find "$BASE/lite_runs" -maxdepth 1 -mindepth 1 -type d -name "${case_name}_*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: no run directory found for $case_name" >&2
    return 1
  fi

  echo "============================================================"
  echo "case=$case_name"
  echo "run_dir=$run_dir"
  echo "results=$RESULTS_ROOT/$case_name"
  echo "start=$(date --iso-8601=seconds)"

  make -C "$ANALYSIS_DIR" particles \
    DATA_DIR="$run_dir" \
    CASE="$case_name" \
    PYTHON=python3 \
    RESULTS_ROOT="$RESULTS_ROOT"

  echo "end=$(date --iso-8601=seconds)"
  find "$RESULTS_ROOT/$case_name/03_particles" -type f \
    \( -name "*.png" -o -name "*.csv" -o -name "*.json" \) \
    -printf "%p\n" | sort
}

run_case M_lite
run_case F_lite
run_case W_lite

echo "all_done=$(date --iso-8601=seconds)"
echo "RESULTS_ROOT=$RESULTS_ROOT"
