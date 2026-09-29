#!/bin/bash
#
# =====================================================================
#  SLURM job: psc_mirror_bikappa3_isotropic — COSMA7-rp
#
#  Isotropic control of psc_mirror_bikappa3_moderate: bi-kappa (kappa=3),
#  A_i = 1 with the same ion thermal energy (beta_i_parallel = 25/3),
#  beta_e_parallel = 1, mass_ratio = 200, grid 576x576, 1000 ppc -- the
#  numerics, cadence and decomposition of its twin. It is mirror stable,
#  so the heating of its electrons is the numerical heating of the setup,
#  which CodeforAnalisys/energy_audit.py subtracts from the anisotropic
#  run. PSC_ENERGIES_EVERY=500 is required: the subtraction uses diag.asc.
#
#  Build the executable once:
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_mirror_bikappa3_isotropic \
#      src/cosma_build_psc_adios2.sh
#
#  Submit:
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    sbatch cosma_jobs/simulacion/sim_mirror_bikappa3_isotropic.sh
#
#  Resume from a checkpoint (same folder):
#    sbatch --export=ALL,RUN_TAG=<jobid>,PSC_RESTART=<checkpoint>.bp <this script>
# =====================================================================

#SBATCH --job-name=psc_mirror_bikappa3_iso
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=37
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=1024
#SBATCH --time=48:00:00
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
# The single valid build directory is "build" (src/ADIOS2_COSMA_RUNBOOK.md).
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET=psc_mirror_bikappa3_isotropic
# Folder by RUN_TAG (not SLURM_JOB_ID), so that a PSC_RESTART resume
# writes into the SAME folder.
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${RUN_TAG}"

# Resolution and cadence of psc_mirror_bikappa3_moderate; do not change one
# without the other, or the control no longer measures the twin's heating.
PSC_NGRID="${PSC_NGRID:-576}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-32}"
PSC_NP_Z="${PSC_NP_Z:-32}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-500}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_CHECKPOINT_EVERY PSC_ENERGIES_EVERY PSC_LAUNCHER

# PSC_RESTART is exported only when given: the case then starts from that
# checkpoint instead of t = 0.
if [ -n "${PSC_RESTART:-}" ]; then
  export PSC_RESTART
fi

# shellcheck source=src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

test -x "$BUILD_DIR/src/$PSC_TARGET" || {
  echo "ERROR: executable not found: $BUILD_DIR/src/$PSC_TARGET" >&2
  echo "       Run: cd $REPO && BUILD_DIR=\"\$PWD/build\" BUILD_JOBS=4 \\" >&2
  echo "            PSC_TARGETS=$PSC_TARGET src/cosma_build_psc_adios2.sh" >&2
  exit 1
}

# Submitting without PSC_RESTART onto a folder that already has data would
# restart from zero on top of an existing run: abort.
if [ -z "${PSC_RESTART:-}" ] && compgen -G "$RUN_DIR/pfd.*" >/dev/null 2>&1; then
  echo "ERROR: $RUN_DIR already contains snapshots and PSC_RESTART was not given." >&2
  echo "       To resume:  --export=ALL,RUN_TAG=$RUN_TAG,PSC_RESTART=<checkpoint>.bp" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp "$BUILD_DIR/src/$PSC_TARGET" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

echo "target=$PSC_TARGET"
echo "job=$SLURM_JOB_ID"
echo "run_tag=$RUN_TAG   (resume with RUN_TAG=$RUN_TAG)"
echo "restart=${PSC_RESTART:-<none, from t=0>}"
echo "nodes=$SLURM_JOB_NODELIST"
echo "ntasks=$SLURM_NTASKS"
echo "ngrid=$PSC_NGRID"
echo "nicell=$PSC_NICELL"
echo "np=1x${PSC_NP_Y}x${PSC_NP_Z}"
echo "energies_every=$PSC_ENERGIES_EVERY"
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "adios2_config=$(command -v adios2-config)"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
