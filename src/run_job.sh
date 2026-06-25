#!/bin/bash
# Submit both turbulence cases as independent jobs:
#   sbatch src/run_job.sh
#
# Submit only one case:
#   sbatch --array=0 src/run_job.sh  # kappa
#   sbatch --array=1 src/run_job.sh  # Maxwellian

#SBATCH --job-name=psc_turb3d
#SBATCH --array=0-1
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=1120
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=/cosma7/data/dp433/dc-mart18/%x_%A_%a.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jeffersson.agudelo@northumbria.ac.uk

set -euo pipefail

BASE="${BASE:-/cosma7/data/dp433/dc-mart18}"
REPO="${REPO:-$BASE/pcseditado}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-adios2-nohdf5}"
RUN_ROOT="${RUN_ROOT:-$BASE/turbulence_3D}"

TARGETS=(
  psc_turbulence_3D_kappa_3
  psc_turbulence_3D_maxwellian_3
)

array_index="${SLURM_ARRAY_TASK_ID:-0}"
if (( array_index < 0 || array_index >= ${#TARGETS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be 0 or 1" >&2
  exit 2
fi
TARGET="${TARGETS[$array_index]}"

# shellcheck source=src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

EXECUTABLE="$BUILD_DIR/src/$TARGET"
test -x "$EXECUTABLE" || {
  echo "ERROR: executable not found: $EXECUTABLE" >&2
  echo "Run first: BUILD_JOBS=8 $REPO/src/cosma_build_psc_adios2.sh" >&2
  exit 1
}

RUN_DIR="$RUN_ROOT/${TARGET}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/logs" "$RUN_DIR"
cp "$EXECUTABLE" "$RUN_DIR/"
if [ -f "$REPO/adios2cfg.xml" ]; then
  cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
fi
cd "$RUN_DIR"

echo "target=$TARGET"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "ntasks=$SLURM_NTASKS"
echo "run_dir=$RUN_DIR"
echo "start=$(date --iso-8601=seconds)"

psc_mpi_run "$SLURM_NTASKS" "./$TARGET"

echo "end=$(date --iso-8601=seconds)"
