#!/bin/bash
#
# =====================================================================
#  Job SLURM: psc_whistler_bikappa5_strong_mr800 — COSMA7-rp
#
#  Whistler Strong Bi-Kappa 5 (kappa=5.0), VARIANTE mi/me = 800: beta_e_par=0.5,
#  Ae=Te_perp/Te_par=3.0, beta_i_par=1.0, Ai=1.0. Identico a
#  psc_whistler_bikappa5_strong salvo PSC_MASS_RATIO (compile-time, por
#  eso es otro ejecutable). Justificacion: src/WHISTLER_PARAMETROS.md.
#
#  Que cambia con mi/me = 800 y que no:
#    - d_i = 28.3 d_e (antes 14.1): la caja de 20 d_i mide 566 d_e, asi
#      que para conservar la resolucion electronica (dx = 0.245 d_e,
#      dx/lambda_De = 6.1) la grilla pasa a 2304^2. nicell 2000.
#    - B0, betas, anisotropias, omega_pe/Omega_ce = 12.5 y dt (0.165
#      wpe^-1, 75.8 pasos por Omega_ce^-1) NO cambian.
#    - La fisica del whistler tampoco: teoria lineal con mi/me=800 da
#      gamma = 0.175 / 0.161 / 0.145 Omega_ce (bi-Max / k5 / k3), igual
#      que con 200 a <0.5%. Por eso nmax y cadencias en Omega_ce son las
#      mismas; lo que cambia es la separacion de escalas: la corrida
#      entera dura 0.66 Omega_ci^-1 (iones practicamente quietos).
#
#  Duracion y salidas:
#    - nmax 40000 = 528 Omega_ce^-1: SOLO hasta la relajacion (criterio
#      t_fin ~ 8 x t_10 del gemelo mas lento, k3: t_10 = 69 Omega_ce^-1).
#      Si A_e(t) sigue bajando al final, se EXTIENDE desde el checkpoint
#      final con un PSC_NMAX mayor.
#    - fields cada 100 = 1.32 Omega_ce^-1 -> Nyquist 2.38 Omega_ce. 400
#      snapshots pfd + 400 pfd_moments (~0.30 TB).
#    - particles cada 5000 = 66 Omega_ce^-1 -> 9 dumps (pasos 0..40000,
#      ~30 GB c/u, ~0.28 TB). Menos que con mi/me=200 por la cuota del
#      grupo; A_e(t) denso sale de los momentos, los dumps dan kappa_eff.
#    - energies cada 20 = 0.26 Omega_ce^-1.
#    - checkpoint SOLO el final (PSC_CHECKPOINT_EVERY = nmax), ~0.64 TB
#      (~30 B/particula medido en COSMA). Conservarlo hasta confirmar la
#      relajacion; si el job muere a medio camino se relanza desde t=0.
#  Presupuesto (calibrado con las corridas ionicas de 48 h en 1024
#  ranks): 2.1e10 particulas x 40000 pasos -> ~23 h en 2304 ranks / 83
#  nodos (~63k core-h por corrida, ~190k la serie). RAM ~0.7-1.4 TB en
#  total, ~8-16 GB por nodo de 510 GB. Disco: ~0.58 TB durable +
#  ~0.64 TB de checkpoint por corrida.
#  IMPORTANTE (paridad): los tres gemelos mr800 (bi-Max, k3, k5)
#  comparten TODOS estos valores.
#
#  Antes de enviar, compilar el ejecutable (una sola vez):
#    cd /cosma7/data/dp433/dc-mart18/pcseditado
#    BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
#      PSC_TARGETS=psc_whistler_bikappa5_strong_mr800 \
#      src/cosma_build_psc_adios2.sh
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/simulacion/sim_whistler_bikappa5_strong_mr800.sh
#
#  EXTENDER una corrida terminada (PSC_NMAX es el paso final ABSOLUTO):
#    sbatch --export=ALL,RUN_TAG=<tag>,PSC_NMAX=60000,PSC_RESTART=<run_dir>/checkpoint_40000.bp \
#      cosma_jobs/simulacion/sim_whistler_bikappa5_strong_mr800.sh
# =====================================================================

#SBATCH --job-name=psc_whistler_bikappa5_strong_mr800
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
PSC_TARGET=psc_whistler_bikappa5_strong_mr800

# =====================================================================
#  Carpeta de la corrida: se identifica por RUN_TAG, no por
#  SLURM_JOB_ID. Asi una reanudacion escribe en la MISMA carpeta y la
#  corrida queda entera en un solo sitio. Sin RUN_TAG se usa el job id
#  del primer envio, que queda impreso abajo para poder reanudar.
# =====================================================================
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${RUN_TAG}"

# mi/me = 800: la caja de 20 d_i mide 566 d_e; 2304^2 conserva la
# resolucion electronica de la familia whistler (dx = 0.245 d_e).
# np 48x48 -> parches de 48x48 celdas. Identico en los tres gemelos mr800.
PSC_NGRID="${PSC_NGRID:-2304}"
PSC_NICELL="${PSC_NICELL:-2000}"
PSC_NP_Y="${PSC_NP_Y:-48}"
PSC_NP_Z="${PSC_NP_Z:-48}"
# Duracion y cadencias de escala electronica (ver cabecera y
# src/WHISTLER_PARAMETROS.md). La duracion es propia de cada regimen
# (solo hasta la relajacion); identica entre gemelos de distribucion.
PSC_NMAX="${PSC_NMAX:-40000}"
PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-100}"
PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-5000}"
PSC_ENERGIES_EVERY="${PSC_ENERGIES_EVERY:-20}"
PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-40000}"
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

# PSC abre un mem-<rank>.log por cada rank MPI (include/psc.hxx) pero solo
# escribe en el con CUDA: en CPU quedan miles de archivos vacios (uno por
# rank). Se borran al salir, tambien si el job termina por walltime.
trap 'find "$RUN_DIR" -maxdepth 1 -name "mem-*.log" -empty -delete 2>/dev/null || true' EXIT

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
