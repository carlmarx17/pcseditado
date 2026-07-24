#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_firehose_bimaxwellian_moderate_bigbox40 — COSMA7-rp
#
#  Firehose Moderate Bi-Maxwelliana en caja de 40 d_i (el doble de la
#  estandar de 20 d_i). Misma fisica que
#  psc_firehose_bimaxwellian_moderate (beta_i_par=6.0, Ai=0.3,
#  mass_ratio=200, 1000 nicell). Con la caja el doble de grande,
#  ngrid=1152 mantiene la MISMA resolucion que los casos estandar
#  (~28.8 celdas/d_i). Coste ~4x un caso estandar; se compensa con 4x
#  ranks (np 64x64, parches de 18x18 celdas, identicos a los de 20 d_i).
#
#  IMPORTANTE: el tamano de caja NO es una variable de entorno; esta
#  fijo en compile-time (#define PSC_DOMAIN_DI en
#  src/psc_anisotropy_case.hxx). Por eso hay un ejecutable dedicado
#  (src/psc_firehose_bimaxwellian_moderate_bigbox40.cxx) que hay que
#  COMPILAR una sola vez antes de enviar:
#
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_firehose_bimaxwellian_moderate_bigbox40 \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/simulacion/sim_firehose_bimaxwellian_moderate_40di.sh
# =====================================================================

#SBATCH --job-name=psc_firehose_bimax_mod_40di
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=147
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=4096
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
PSC_TARGET=psc_firehose_bimaxwellian_moderate_bigbox40
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"

# ngrid=1152 en caja de 40 d_i = misma resolucion que los casos
# estandar (576 en 20 d_i, ~28.8 celdas/d_i). La caja de 40 d_i vive
# en el ejecutable (PSC_DOMAIN_DI=40), no aqui.
PSC_NGRID="${PSC_NGRID:-1152}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-64}"
PSC_NP_Z="${PSC_NP_Z:-64}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-0}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_CHECKPOINT_EVERY PSC_ENERGIES_EVERY PSC_LAUNCHER

# shellcheck source=../../src/cosma_adios2_env.sh
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
echo "domain_di=40  (compile-time)"
echo "ngrid=$PSC_NGRID  -> ~$(awk "BEGIN{printf \"%.1f\", $PSC_NGRID/40.0}") celdas/d_i"
echo "nicell=$PSC_NICELL"
echo "np=1x${PSC_NP_Y}x${PSC_NP_Z}"
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
