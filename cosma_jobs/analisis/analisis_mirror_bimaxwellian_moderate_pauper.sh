#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: analysis_mirror_bimaxwellian_moderate — COSMA7-rp-pauper
#
#  Corre el pipeline de analisis Python (CodeforAnalisys/Makefile,
#  target "all" = "analysis") sobre una corrida ya finalizada. Las 8
#  etapas independientes del target "common" (brazil, fields, particles,
#  spectral, diamagnetic, heatflux, validate, physics) no dependen entre
#  si, asi que se reparten en 8 nodos separados via srun en lugar de
#  correr secuencialmente en un solo nodo.
#
#  Envio: sbatch cosma_jobs/analisis/analisis_mirror_bimaxwellian_moderate_pauper.sh
# =====================================================================

# --- Identificacion del job ---
#SBATCH --job-name=an_mirror_mod

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_mirror_moderate.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_mirror_moderate.%J.err

# --- Particion y cuenta ---
# cosma7-rp-pauper: misma piscina de nodos que cosma7-rp, cola de bajo
# costo, limite de tiempo 24h. Cambiar a cosma7-rp (72h) si hace falta.
#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433

# --- Recursos ---
# 8 nodos, 1 tarea por nodo, nodo completo (28 cores) por tarea: cada
# etapa del pipeline es un proceso Python (no MPI) que puede usar
# internamente varios cores (ProcessPoolExecutor) para su propio trabajo.
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# --- Tiempo maximo ---
# El analisis es mucho mas corto que una corrida PSC; ajustar si el
# dataset es muy grande (maximo permitido por la particion: 24h).
#SBATCH --time=08:00:00

# --- Directorio de trabajo ---
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys

# --- Notificacion por correo ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

# =====================================================================
# Entorno: mismos modulos + variables ADIOS2/HDF5 que
# src/submit_anisotropy_adios2.slurm, para que los scripts de Python
# puedan leer snapshots ADIOS2 (.bp) si la corrida quedo en ese formato.
# =====================================================================
REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

cd "$REPO/CodeforAnalisys"

DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_bimaxwellian_moderate_11596993
CASE=mirror_bimaxwellian_moderate
RESULTS_ROOT=../analysis_results

mkdir -p /cosma7/data/dp433/dc-mart18/logs

# =====================================================================
# El Makefile solo reconoce el patron .h5 en sus guardas check-fields /
# check-moments / check-particles (los scripts de Python si soportan
# .h5 y .bp via data_reader.py). Si la corrida solo dejo snapshots
# ADIOS2 (.bp), hay que forzar los patrones explicitamente o esas
# guardas fallan aunque los datos existan.
# =====================================================================
EXTRA_VARS=()
if ! compgen -G "$DATA_DIR/pfd.*_p*.h5" > /dev/null && \
   compgen -G "$DATA_DIR/pfd.*.bp" > /dev/null; then
    echo "Formato detectado: ADIOS2 (.bp) -> forzando FIELD/MOMENT/PARTICLE_PATTERN"
    EXTRA_VARS=(
        "FIELD_PATTERN=$DATA_DIR/pfd.*.bp"
        "MOMENT_PATTERN=$DATA_DIR/pfd_moments.*.bp"
        "PARTICLE_PATTERN=$DATA_DIR/prt_${CASE}.*.bp"
    )
else
    echo "Formato detectado: HDF5 (.h5)"
fi

echo "============================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Nodos       : $SLURM_JOB_NODELIST"
echo " DATA_DIR    : $DATA_DIR"
echo " CASE        : $CASE"
echo " Inicio      : $(date)"
echo "============================================"

# =====================================================================
# 1) manifest: rapido, corre primero en un solo nodo de la asignacion
# =====================================================================
srun --nodes=1 --ntasks=1 --exclusive \
    make manifest DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${EXTRA_VARS[@]}"
manifest_rc=$?
if [ "$manifest_rc" -ne 0 ]; then
    echo "ERROR: 'make manifest' fallo (rc=$manifest_rc); abortando." >&2
    exit "$manifest_rc"
fi

# =====================================================================
# 2) Las 8 etapas independientes de "common", una por nodo, en paralelo
# =====================================================================
STAGES=(brazil fields particles spectral diamagnetic heatflux validate physics)
declare -A PIDS

for stage in "${STAGES[@]}"; do
    LOG="/cosma7/data/dp433/dc-mart18/logs/analysis_${CASE}_${stage}.${SLURM_JOB_ID}.log"
    srun --nodes=1 --ntasks=1 --exclusive --job-name="$stage" \
        make "$stage" DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${EXTRA_VARS[@]}" \
        > "$LOG" 2>&1 &
    PIDS[$stage]=$!
done

fail=0
for stage in "${STAGES[@]}"; do
    if wait "${PIDS[$stage]}"; then
        echo "OK: etapa '$stage' completada."
    else
        echo "ERROR: etapa '$stage' fallo. Ver /cosma7/data/dp433/dc-mart18/logs/analysis_${CASE}_${stage}.${SLURM_JOB_ID}.log" >&2
        fail=1
    fi
done

echo "============================================"
echo " Fin: $(date)"
echo "============================================"
exit "$fail"
