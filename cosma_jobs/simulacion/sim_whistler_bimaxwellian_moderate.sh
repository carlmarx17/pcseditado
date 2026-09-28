#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_whistler_bimaxwellian_moderate — COSMA7-rp
#
#  Whistler moderado (beta_e_parallel=0.5, Ae=2.0, iones isotropos
#  beta_i_parallel=1.0), distribucion inicial bi-Maxwelliana.
#  Gemelo de distribucion: psc_whistler_bikappa3_moderate. Los dos scripts deben tener
#  EXACTAMENTE las mismas variables PSC_* (paridad de la comparacion).
#
#  Cadencia y duracion propias de la familia whistler (escala electronica),
#  distintas de mirror/firehose a proposito. En unidades del codigo
#  (576^2 en 20 d_i): dt = 0.3299 / omega_pe, Omega_ce dt = 0.02639,
#  Omega_ci dt = 1.32e-4. Teoria lineal (linear_theory.py, k || B0):
#    bi-Maxwellian  gamma_max = 0.054 Omega_ce, omega_r = 0.39 Omega_ce, k d_e = 0.71
#    bi-Kappa k=3   gamma_max = 0.044 Omega_ce, omega_r = 0.37 Omega_ce, k d_e = 0.69
#  e-folding ~ 19-23 / Omega_ce; saturacion esperada en Omega_ce t ~ 150-250.
#
#    PSC_NMAX=80000           -> Omega_ce t = 2111 (Omega_ci t = 10.6):
#                                ~10x el tiempo de saturacion, cubre la
#                                relajacion cuasi-lineal de Ae.
#    PSC_FIELDS_EVERY=100     -> Delta t Omega_ce = 2.64, Nyquist 1.19 Omega_ce
#                                (toda la rama whistler 0 < omega < Omega_ce sin
#                                aliasing; ~7 muestras por e-folding). 800 snapshots.
#    PSC_ENERGIES_EVERY=20    -> Delta t Omega_ce = 0.53, ajuste fino de gamma.
#    PSC_PARTICLES_EVERY=2000 -> Delta t Omega_ce = 52.8, 41 VDFs para A_e(t)
#                                y kappa_eff(t).
#    PSC_CHECKPOINT_EVERY=40000 -> un checkpoint a mitad de corrida.
#
#  Antes de enviar, compilar el ejecutable (una sola vez):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS="psc_whistler_bimaxwellian_moderate psc_whistler_bikappa3_moderate" \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (los dos gemelos juntos):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    sbatch cosma_jobs/simulacion/sim_whistler_bimaxwellian_moderate.sh
#    sbatch cosma_jobs/simulacion/sim_whistler_bikappa3_moderate.sh
# =====================================================================

#SBATCH --job-name=psc_whistler_bimax_mod
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=37
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=1024
#SBATCH --time=12:00:00
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
# Un unico directorio de build valido: "build" (ver
# src/ADIOS2_COSMA_RUNBOOK.md). No usar build-adios2-nohdf5.
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET=psc_whistler_bimaxwellian_moderate
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"

# Misma resolucion que el resto de la matriz (576^2, 1000 ppc); duracion y
# cadencia de salida de la familia whistler (ver encabezado). Identicas en
# los dos gemelos whistler moderate.
PSC_NGRID="${PSC_NGRID:-576}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-32}"
PSC_NP_Z="${PSC_NP_Z:-32}"
PSC_NMAX="${PSC_NMAX:-80000}"
PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-100}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-20}"
PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-2000}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-40000}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_NMAX PSC_FIELDS_EVERY \
  PSC_ENERGIES_EVERY PSC_PARTICLES_EVERY PSC_CHECKPOINT_EVERY PSC_LAUNCHER

# shellcheck source=src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

test -x "$BUILD_DIR/src/$PSC_TARGET" || {
  echo "ERROR: executable not found: $BUILD_DIR/src/$PSC_TARGET" >&2
  echo "       Run: cd $REPO && BUILD_DIR=\"\$PWD/build\" BUILD_JOBS=4 \\" >&2
  echo "            PSC_TARGETS=$PSC_TARGET src/cosma_build_psc_adios2.sh" >&2
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
echo "energies_every=$PSC_ENERGIES_EVERY"
echo "nmax=$PSC_NMAX"
echo "fields_every=$PSC_FIELDS_EVERY"
echo "particles_every=$PSC_PARTICLES_EVERY"
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "adios2_config=$(command -v adios2-config)"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
