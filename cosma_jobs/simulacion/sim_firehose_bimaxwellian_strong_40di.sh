#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_firehose_bimaxwellian_strong_bigbox40 — COSMA7-rp
#
#  Firehose Strong Bi-Maxwelliano en caja de 40 d_i. Es el GEMELO
#  CONTROLADO de psc_firehose_bikappa3_bigbox40: mismo beta_i_par=10.0,
#  misma Ai=0.1, mismo mass_ratio=200, mismo nicell=1000, misma caja y
#  misma grilla. Lo unico que cambia entre los dos es la distribucion
#  (Bi-Maxwelliana vs Bi-Kappa kappa=3), que es justo lo que se quiere
#  aislar.
#
#  Por que hace falta: el par que existia
#  (firehose_bimaxwellian_moderate_bigbox40 vs firehose_bikappa3_bigbox40)
#  difiere en beta_i_par (6 vs 10) Y en Ai (0.3 vs 0.1), asi que su
#  comparacion mezcla el efecto de kappa con el de la energia libre
#  disponible. Con esta corrida el par queda limpio.
#
#  Coste ~4x un caso estandar; np 48x48 (2304 ranks, 83 nodos), igual
#  que el bikappa3 bigbox40 para que el coste numerico sea comparable.
#
#  IMPORTANTE: el tamano de caja NO es una variable de entorno; esta
#  fijo en compile-time (#define PSC_DOMAIN_DI en
#  src/psc_anisotropy_case.hxx). Por eso hay un ejecutable dedicado
#  (src/psc_firehose_bimaxwellian_strong_bigbox40.cxx) que hay que
#  COMPILAR una sola vez antes de enviar:
#
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_firehose_bimaxwellian_strong_bigbox40 \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/simulacion/sim_firehose_bimaxwellian_strong_40di.sh
#
#  REANUDAR si las 48 h no alcanzan (NO reenviar sin esto: un reenvio
#  pelado empieza otra corrida desde t=0 en una carpeta nueva, que es
#  como el bikappa3 bigbox40 acabo partido en varias carpetas):
#    sbatch --export=ALL,RUN_TAG=<tag-de-la-corrida>,PSC_RESTART=/ruta/checkpoint_<step>.bp \
#      cosma_jobs/simulacion/sim_firehose_bimaxwellian_strong_40di.sh
# =====================================================================

#SBATCH --job-name=psc_firehose_bimaxwellian_strong_40di
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=83
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=2304
#SBATCH --exclude=m7010,m7174
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
PSC_TARGET=psc_firehose_bimaxwellian_strong_bigbox40

# =====================================================================
#  Carpeta de la corrida: se identifica por RUN_TAG, no por
#  SLURM_JOB_ID. Asi una reanudacion escribe en la MISMA carpeta y la
#  corrida queda entera en un solo sitio. Sin RUN_TAG se usa el job id
#  del primer envio, que queda impreso abajo para poder reanudar.
# =====================================================================
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${RUN_TAG}"

# ngrid=1152 en caja de 40 d_i = misma resolucion que los casos
# estandar (576 en 20 d_i, ~28.8 celdas/d_i). La caja de 40 d_i vive
# en el ejecutable (PSC_DOMAIN_DI=40), no aqui.
PSC_NGRID="${PSC_NGRID:-1152}"
PSC_NICELL="${PSC_NICELL:-1000}"
PSC_NP_Y="${PSC_NP_Y:-48}"
PSC_NP_Z="${PSC_NP_Z:-48}"
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
echo "domain_di=40  (compile-time)"
echo "ngrid=$PSC_NGRID  -> ~$(awk "BEGIN{printf \"%.1f\", $PSC_NGRID/40.0}") celdas/d_i"
echo "nicell=$PSC_NICELL"
echo "np=1x${PSC_NP_Y}x${PSC_NP_Z}"
echo "run_dir=$RUN_DIR"
echo "restart=${PSC_RESTART:-<none, desde t=0>}"
echo "adios2_dir=$ADIOS2_DIR"
echo "launcher=${PSC_LAUNCHER:-srun}"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true
module list 2>&1

psc_mpi_run "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
