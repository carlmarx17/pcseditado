#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_whistler_bimaxwellian_weak — COSMA7-rp
#
#  Whistler Weak Bi-Maxwelliano: beta_e_par=0.5, Ae=Te_perp/Te_par=1.5,
#  beta_i_par=1.0, Ai=1.0, mass_ratio=200. Caja de 20 d_i con la misma
#  grilla, ppc y descomposicion que los casos mirror estandar
#  (ngrid 576, nicell 1000, np 32x32 = 1024 ranks).
#
#  Checkpoint cada 150000 pasos, igual que mirror y firehose. Sin este
#  script el caso usaba el valor por defecto del header
#  (PSC_CHECKPOINT_EVERY_DEFAULT = 5000) y dejaba 240 checkpoints por
#  corrida. Los checkpoints solo sirven para reanudar; el analisis no
#  los lee.
#
#  Antes de enviar, compilar el ejecutable (una sola vez):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_whistler_bimaxwellian_weak \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/simulacion/sim_whistler_bimaxwellian_weak.sh
#
#  REANUDAR si las 72 h no alcanzan (NO reenviar sin esto: un reenvio
#  pelado empieza otra corrida desde t=0 en una carpeta nueva):
#    sbatch --export=ALL,RUN_TAG=<tag-de-la-corrida>,PSC_RESTART=/ruta/checkpoint_<step>.bp \
#      cosma_jobs/simulacion/sim_whistler_bimaxwellian_weak.sh
# =====================================================================

#SBATCH --job-name=psc_whistler_bimaxwellian_weak
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=37
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=1024
#SBATCH --time=72:00:00
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET=psc_whistler_bimaxwellian_weak

# =====================================================================
#  Carpeta de la corrida: se identifica por RUN_TAG, no por
#  SLURM_JOB_ID. Asi una reanudacion escribe en la MISMA carpeta y la
#  corrida queda entera en un solo sitio. Sin RUN_TAG se usa el job id
#  del primer envio, que queda impreso abajo para poder reanudar.
# =====================================================================
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${RUN_TAG}"

# Misma resolucion/cadencia que los casos mirror estandar.
PSC_NGRID="${PSC_NGRID:-576}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-32}"
PSC_NP_Z="${PSC_NP_Z:-32}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-500}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_CHECKPOINT_EVERY PSC_ENERGIES_EVERY PSC_LAUNCHER

# PSC_RESTART solo se exporta si viene definido: el caso lo lee con
# getenv y arranca desde ese checkpoint en vez de desde t=0.
if [ -n "${PSC_RESTART:-}" ]; then
  export PSC_RESTART
fi

# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

test -x "$BUILD_DIR/src/$PSC_TARGET" || {
  echo "ERROR: executable not found: $BUILD_DIR/src/$PSC_TARGET" >&2
  echo "       Run: cd $REPO && BUILD_DIR=\"\$PWD/build\" BUILD_JOBS=4 \\" >&2
  echo "            PSC_TARGETS=$PSC_TARGET src/cosma_build_psc_adios2.sh" >&2
  exit 1
}

# Un envio sin PSC_RESTART sobre una carpeta que ya tiene datos
# empezaria de cero encima de una corrida existente. Se aborta.
if [ -z "${PSC_RESTART:-}" ] && compgen -G "$RUN_DIR/pfd.*" >/dev/null 2>&1; then
  echo "ERROR: $RUN_DIR ya contiene snapshots y no se paso PSC_RESTART." >&2
  echo "       Para reanudar:  --export=ALL,RUN_TAG=$RUN_TAG,PSC_RESTART=<checkpoint>.bp" >&2
  echo "       Para empezar otra corrida distinta: usa otro RUN_TAG." >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp "$BUILD_DIR/src/$PSC_TARGET" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

echo "target=$PSC_TARGET"
echo "job=$SLURM_JOB_ID"
echo "run_tag=$RUN_TAG   (reanuda con RUN_TAG=$RUN_TAG)"
echo "nodes=$SLURM_JOB_NODELIST"
echo "ntasks=$SLURM_NTASKS"
echo "ngrid=$PSC_NGRID"
echo "nicell=$PSC_NICELL"
echo "np=1x${PSC_NP_Y}x${PSC_NP_Z}"
echo "checkpoint_every=$PSC_CHECKPOINT_EVERY"
echo "energies_every=$PSC_ENERGIES_EVERY"
echo "run_dir=$RUN_DIR"
echo "restart=${PSC_RESTART:-<none, desde t=0>}"
echo "adios2_dir=$ADIOS2_DIR"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
