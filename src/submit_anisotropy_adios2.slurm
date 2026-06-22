#!/bin/bash
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1024
#SBATCH --time=48:00:00
#SBATCH --job-name=psc_aniso
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="$REPO/build-adios2-nohdf5"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET="${PSC_TARGET:-psc_mirror_kappa3}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"

# shellcheck source=src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

test -x "$BUILD_DIR/src/$PSC_TARGET" || {
  echo "ERROR: executable not found: $BUILD_DIR/src/$PSC_TARGET" >&2
  echo "       Run: BUILD_JOBS=4 src/cosma_build_psc_adios2.sh" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
cp "$BUILD_DIR/src/$PSC_TARGET" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

echo "target=$PSC_TARGET"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "ntasks=$SLURM_NTASKS"
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "adios2_config=$(command -v adios2-config)"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true

export PSC_NICELL="${PSC_NICELL:-2000}"
export PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-7500}"
export PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-1000}"
export PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-10000}"

mpirun -np "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
