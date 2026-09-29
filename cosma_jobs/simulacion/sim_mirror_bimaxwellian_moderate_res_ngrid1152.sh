#!/bin/bash
#
# =====================================================================
#  SLURM job: resolution test "ngrid1152" of psc_mirror_bimaxwellian_moderate
#  — COSMA7-rp
#
#  Same executable, physics and box as the production run; only
#  half the cell size (PSC_NGRID=1152, dx/lambda_De = 4.3), same ppc; dt halves by CFL. The run stops at t Omega_ci ~ 41 (linear phase and onset
#  of saturation), with the production output cadence in physical time.
#  Purpose: identify the numerical electron heating (x8 by t Omega_ci =
#  158, T_e/T_e0 = 1.45 at t Omega_ci = 20, see
#  CodeforAnalisys/IMPLEMENTATION_V6.md) and test convergence of gamma.
#  A grid (finite-grid) heating falls steeply with dx/lambda_De;
#  noise heating per unit time changes little.
#
#  Build (once; the production executable, nothing new):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_mirror_bimaxwellian_moderate \
#      src/cosma_build_psc_adios2.sh
#
#  Submit:
#    sbatch cosma_jobs/simulacion/sim_mirror_bimaxwellian_moderate_res_ngrid1152.sh
#
#  Resume from a checkpoint (same folder):
#    sbatch --export=ALL,RUN_TAG=<jobid>,PSC_RESTART=<checkpoint>.bp <this script>
#
#  Analysis: cosma_jobs/analisis/analisis_resolution_mirror.sh
# =====================================================================

#SBATCH --job-name=psc_mirror_res_ngrid1152
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=83
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=2304
#SBATCH --time=48:00:00
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET=psc_mirror_bimaxwellian_moderate
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
# Own folder, so the variant is never mixed with the production run.
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_res_ngrid1152_${RUN_TAG}"

# The variant: everything else is the header default of the production run.
# 48x48 patches (24x24 cells) on 2304 ranks, as the 1152^2 whistler jobs.
PSC_NGRID="${PSC_NGRID:-1152}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-48}"
PSC_NP_Z="${PSC_NP_Z:-48}"
PSC_NMAX="${PSC_NMAX:-620000}"
# Output cadence in steps chosen for the production cadence in physical time.
PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-1000}"
PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-20000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-1000}"
PSC_CONTINUITY_EVERY="${PSC_CONTINUITY_EVERY:-10000}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-300000}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_NMAX PSC_FIELDS_EVERY PSC_PARTICLES_EVERY \
       PSC_ENERGIES_EVERY PSC_CONTINUITY_EVERY PSC_CHECKPOINT_EVERY PSC_LAUNCHER

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

if [ -z "${PSC_RESTART:-}" ] && compgen -G "$RUN_DIR/pfd.*" >/dev/null 2>&1; then
  echo "ERROR: $RUN_DIR already contains snapshots and PSC_RESTART was not given." >&2
  echo "       To resume:  --export=ALL,RUN_TAG=$RUN_TAG,PSC_RESTART=<checkpoint>.bp" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp "$BUILD_DIR/src/$PSC_TARGET" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

# Runtime values for the analysis (run_pipeline.py reads this file, so dt,
# ppc, nmax and cadences are measured facts, not the production profile).
# dt = CFL 0.95 * dx / sqrt(2), dx = 20 d_i / ngrid, d_i = sqrt(200) d_e.
DT_CODE=$(awk -v n="$PSC_NGRID" 'BEGIN { printf "%.12g", 0.95 * 20 * sqrt(200) / n / sqrt(2) }')
cat > analysis_config.json <<JSON
{"ngrid": $PSC_NGRID, "nicell": $PSC_NICELL, "nmax": $PSC_NMAX, "dt_code": $DT_CODE,
 "fields_every": $PSC_FIELDS_EVERY, "particles_every": $PSC_PARTICLES_EVERY}
JSON

echo "target=$PSC_TARGET  variant=ngrid1152"
echo "job=$SLURM_JOB_ID  run_tag=$RUN_TAG  restart=${PSC_RESTART:-<none, from t=0>}"
echo "ngrid=$PSC_NGRID  nicell=$PSC_NICELL  np=1x${PSC_NP_Y}x${PSC_NP_Z}  nmax=$PSC_NMAX  dt=$DT_CODE"
echo "fields_every=$PSC_FIELDS_EVERY  particles_every=$PSC_PARTICLES_EVERY  energies_every=$PSC_ENERGIES_EVERY"
echo "run_dir=$RUN_DIR"
echo "start=$(date --iso-8601=seconds)"
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
