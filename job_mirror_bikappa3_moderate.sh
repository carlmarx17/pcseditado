#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_mirror_bikappa3_moderate — COSMA7-rp
#
#  Mismas condiciones fisicas y de resolucion que
#  psc_mirror_bimaxwellian_moderate (beta_i_parallel=5.0, Ai=2.0,
#  mass_ratio=200, grilla 1024x1024, 1500 ppc), pero con distribucion
#  inicial Bi-Kappa (kappa=3) en vez de Bi-Maxwelliana. Pensado para
#  compararse directamente contra la corrida
#  psc_mirror_bimaxwellian_moderate ya existente (mismo Ai/beta).
#
#  Es una copia fijada de src/submit_anisotropy_adios2.slurm con
#  PSC_TARGET=psc_mirror_bikappa3_moderate, siguiendo el procedimiento
#  validado en src/ADIOS2_COSMA_RUNBOOK.md (build unico en build/,
#  launcher mpirun -- no usar srun --mpi=pmi2 con OpenMPI 5).
#
#  Antes de enviar, compilar el ejecutable (una sola vez):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_mirror_bikappa3_moderate \
#      src/cosma_build_psc_adios2.sh
#
#  Envio:
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    sbatch job_mirror_bikappa3_moderate.sh
# =====================================================================

#SBATCH --job-name=psc_mirror_bikappa3_mod
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
# Un unico directorio de build valido: "build" (ver
# src/ADIOS2_COSMA_RUNBOOK.md). No usar build-adios2-nohdf5.
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET=psc_mirror_bikappa3_moderate
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"

# Misma resolucion/cadencia que psc_mirror_bimaxwellian_moderate.
PSC_NGRID="${PSC_NGRID:-576}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-32}"
PSC_NP_Z="${PSC_NP_Z:-32}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-0}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_CHECKPOINT_EVERY PSC_ENERGIES_EVERY PSC_LAUNCHER

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
echo "run_dir=$RUN_DIR"
echo "adios2_dir=$ADIOS2_DIR"
echo "adios2_config=$(command -v adios2-config)"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
