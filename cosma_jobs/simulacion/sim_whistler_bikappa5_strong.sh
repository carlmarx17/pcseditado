#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_whistler_bikappa5_strong — COSMA7-rp
#
#  Whistler Strong Bi-Kappa 5 (kappa=5.0, gemelo de distribucion del
#  bi-Maxwelliano strong): beta_e_par=0.5, Ae=Te_perp/Te_par=3.0,
#  beta_i_par=1.0, Ai=1.0, mass_ratio=200. Caja de 20 d_i como toda la
#  matriz, pero con grilla y ppc REFINADOS solo para la familia whistler
#  (decision registrada en src/WHISTLER_PARAMETROS.md): ngrid 1152
#  (dx = 0.245 d_e, dx/lambda_De = 6.1), nicell 2000, np 48x48 = 2304
#  ranks (misma descomposicion que los jobs bigbox40; parches 24x24).
#
#  El whistler es la unica inestabilidad de escala ELECTRONICA de la
#  matriz (crece en d_e y Omega_ce^-1), asi que la duracion y las
#  cadencias de salida del header (pensadas para escalas ionicas) se
#  pisan aqui por entorno; grilla, caja, ppc y dt quedan identicos al
#  resto de la matriz. Justificacion completa (teoria lineal de los tres
#  regimenes + literatura): src/WHISTLER_PARAMETROS.md. Resumen:
#  Con ngrid 1152 el CFL baja dt a 0.165 wpe^-1 (75.8 pasos por
#  Omega_ce^-1), asi que las cadencias en PASOS se duplican para
#  mantener la misma cadencia FISICA:
#    - nmax 300000 = 3958 Omega_ce^-1 = 19.8 Omega_ci^-1: >=3x el
#      crecimiento del caso debil (gamma_w = 0.0075 Omega_ce) con cola
#      de relajacion.
#    - fields cada 100 pasos = 1.32 Omega_ce^-1 -> Nyquist 2.38
#      Omega_ce (el default del header ALIASA toda la rama whistler,
#      omega_r = 0.25-0.5 Omega_ce). 3000 snapshots (~560 GB
#      campos+momentos por corrida).
#    - particles cada 20000 = 264 Omega_ce^-1 -> 15 dumps de VDF
#      (~8 GB c/u, ~120 GB por corrida).
#    - energies cada 20 = 0.26 Omega_ce^-1 (gamma global gratis).
#    - checkpoint cada 150000 -> 2 por corrida (~340 GB c/u; borrar al
#      terminar con cosma_jobs/utils o cleanup_restart_outputs.sh).
#  Presupuesto (calibrado con las corridas ionicas de 48 h en 1024
#  ranks): 2.0x los pushes de una ionica -> ~43 h en 2304 ranks
#  (~118k core-h por corrida). RAM agregada ~250 GB (~3 GB/nodo).
#  IMPORTANTE (paridad): estos tres scripts whistler deben compartir
#  estos valores; los futuros gemelos bi-kappa whistler los reutilizan
#  tal cual.
#
#  Antes de enviar, compilar el ejecutable (una sola vez):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_whistler_bikappa5_strong \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/simulacion/sim_whistler_bikappa5_strong.sh
#
#  REANUDAR si las 72 h no alcanzan (NO reenviar sin esto: un reenvio
#  pelado empieza otra corrida desde t=0 en una carpeta nueva):
#    sbatch --export=ALL,RUN_TAG=<tag-de-la-corrida>,PSC_RESTART=/ruta/checkpoint_<step>.bp \
#      cosma_jobs/simulacion/sim_whistler_bikappa5_strong.sh
# =====================================================================

#SBATCH --job-name=psc_whistler_bikappa5_strong
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=83
#SBATCH --ntasks-per-node=28
#SBATCH --ntasks=2304
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
PSC_TARGET=psc_whistler_bikappa5_strong

# =====================================================================
#  Carpeta de la corrida: se identifica por RUN_TAG, no por
#  SLURM_JOB_ID. Asi una reanudacion escribe en la MISMA carpeta y la
#  corrida queda entera en un solo sitio. Sin RUN_TAG se usa el job id
#  del primer envio, que queda impreso abajo para poder reanudar.
# =====================================================================
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${RUN_TAG}"

# Grilla y ppc refinados SOLO para la familia whistler (decision
# registrada en src/WHISTLER_PARAMETROS.md; los defaults del header y
# los .cxx siguen en 576/1000). Los tres whistler y sus futuros gemelos
# bi-kappa deben compartir estos valores.
PSC_NGRID="${PSC_NGRID:-1152}"
PSC_NICELL="${PSC_NICELL:-2000}"
PSC_NP_Y="${PSC_NP_Y:-48}"
PSC_NP_Z="${PSC_NP_Z:-48}"
# Duracion y cadencias de escala electronica (ver cabecera y
# src/WHISTLER_PARAMETROS.md). Identicas en los tres casos whistler.
PSC_NMAX="${PSC_NMAX:-300000}"
PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-100}"
PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-20000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-20}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-150000}"
PSC_LAUNCHER="${PSC_LAUNCHER:-mpirun}"
export PSC_NGRID PSC_NICELL PSC_NP_Y PSC_NP_Z PSC_NMAX PSC_FIELDS_EVERY \
       PSC_PARTICLES_EVERY PSC_ENERGIES_EVERY PSC_CHECKPOINT_EVERY PSC_LAUNCHER

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
echo "nmax=$PSC_NMAX"
echo "fields_every=$PSC_FIELDS_EVERY"
echo "particles_every=$PSC_PARTICLES_EVERY"
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
