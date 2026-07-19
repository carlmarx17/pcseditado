#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: analysis_mirror_bimaxwellian_moderate — COSMA7-rp (8 nodos)
#
#  Version de maxima potencia real para este pipeline: identico en
#  diseno a job_analysis_mirror_bimaxwellian_moderate.sh (8 nodos
#  completos, uno por cada una de las 8 etapas independientes de
#  "common": brazil, fields, particles, spectral, diamagnetic,
#  heatflux, validate, physics -> 224 cores en total), pero apuntando
#  a la particion "cosma7-rp" (sin "-pauper") en vez de
#  "cosma7-rp-pauper". Mismo pool fisico de nodos (m70xx-m71xx), solo
#  cambia el tier de prioridad/limite de tiempo.
#
#  Nota: 8 nodos es el techo util de paralelismo de este pipeline --
#  hay exactamente 8 etapas independientes, cada una un proceso Python
#  sin soporte MPI/multi-nodo. Pedir mas nodos que eso no lo hace mas
#  rapido, solo deja nodos reservados sin trabajo asignado.
#
#  Envio: sbatch job_analysis_mirror_bimaxwellian_moderate_cosma7.sh
# =====================================================================

# --- Identificacion del job ---
#SBATCH --job-name=an_mirror_mod_rp

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_mirror_moderate_cosma7.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_mirror_moderate_cosma7.%J.err

# --- Particion y cuenta ---
# cosma7-rp (no pauper): mismo pool fisico que cosma7-rp-pauper (que ya
# usa el job original), pero tier de prioridad mas alto -- utilmente
# distinto solo si se manda junto con el otro job y se quiere que este
# entre primero. Limite de tiempo 72h (vs 24h de -pauper).
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433

# --- Recursos ---
# 8 nodos, 1 tarea por nodo, nodo completo (28 cores) por tarea = 224
# cores en total -- el maximo paralelismo que este pipeline puede usar
# (una etapa independiente por nodo).
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# --- Tiempo maximo ---
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
