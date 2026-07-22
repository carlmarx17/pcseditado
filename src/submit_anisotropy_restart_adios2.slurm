#!/bin/bash
# Restart an anisotropy PSC/ADIOS2 run from an existing checkpoint.
#
# Required:
#   PSC_RESTART=/absolute/path/to/checkpoint_<step>.bp
#
# Optional:
#   PSC_TARGET=psc_mirror_bikappa3_moderate
#   PSC_RUN_DIR=/absolute/path/to/existing/run/dir
#
# If PSC_RUN_DIR is not set, output goes to a new directory and no old files
# are overwritten. If PSC_RUN_DIR is set to an existing run directory, clean
# stale output first with cleanup_restart_outputs.sh.

#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=37
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=1024
#SBATCH --time=48:00:00
#SBATCH --job-name=psc_aniso_restart
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="${BUILD_DIR:-$REPO/build-adios2-nohdf5}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET="${PSC_TARGET:-psc_mirror_bikappa3_moderate}"

: "${PSC_RESTART:?Set PSC_RESTART to an absolute checkpoint_<step>.bp path}"

if [[ "$PSC_RESTART" != /* ]]; then
  echo "ERROR: PSC_RESTART must be an absolute path: $PSC_RESTART" >&2
  exit 2
fi

if [[ ! -d "$PSC_RESTART" ]]; then
  echo "ERROR: checkpoint directory not found: $PSC_RESTART" >&2
  exit 2
fi

RUN_DIR="${PSC_RUN_DIR:-$RUN_ROOT/${PSC_TARGET}_restart_${SLURM_JOB_ID}}"

PSC_NGRID="${PSC_NGRID:-576}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-32}"
PSC_NP_Z="${PSC_NP_Z:-32}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-0}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_CHECKPOINT_EVERY
export PSC_ENERGIES_EVERY PSC_LAUNCHER PSC_RESTART

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
echo "ngrid=$PSC_NGRID"
echo "nicell=$PSC_NICELL"
echo "np=1x${PSC_NP_Y}x${PSC_NP_Z}"
echo "checkpoint_every=$PSC_CHECKPOINT_EVERY"
echo "energies_every=$PSC_ENERGIES_EVERY"
echo "restart=$PSC_RESTART"
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "adios2_config=$(command -v adios2-config)"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
