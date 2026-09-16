#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: re-corrida dirigida de las etapas 'particles' y 'validate'
#  para mirror_bikappa3_moderate — COSMA7-rp-pauper
#
#  Las 8 etapas ya corrieron en el job de analisis original, pero
#  03_particles y 08_validation no quedaron en analysis_results/, o sea
#  que fallaron. Este script re-corre SOLO esas dos en vez de repetir
#  las 8 en 8 nodos.
#
#  Importa porque el par mirror (bimaxwellian ↔ bikappa3) es la
#  comparacion limpia de la matriz: mismo beta, mismo A, misma caja,
#  mismos 1.2M pasos. Si un lado no tiene diagnosticos de particulas ni
#  validacion, cualquier figura comparativa queda coja.
#
#  Antes de enviarlo, revisar por que fallaron:
#    tail -30 $LOGS/analysis_mirror_bikappa3_moderate_particles.*.log
#    ls $DATA_DIR/prt_mirror_bikappa3_moderate.*.bp | head
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/analisis/analisis_mirror_bikappa3_moderate_rerun_particles.sh
# =====================================================================

#SBATCH --job-name=an_bik3_rerun
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_bikappa3_rerun.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_bikappa3_rerun.%J.err

#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433

# Solo 2 etapas -> 2 nodos, una por nodo.
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

#SBATCH --time=06:00:00
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

cd "$REPO/CodeforAnalisys"

DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_bikappa3_moderate_11618877
CASE=mirror_bikappa3_moderate
RESULTS_ROOT="${RESULTS_ROOT:-../analysis_results/run_aware_v4}"
LOGS=/cosma7/data/dp433/dc-mart18/logs

mkdir -p "$LOGS"

# El Makefile solo reconoce .h5 en sus guardas; si la corrida quedo en
# ADIOS2 hay que forzar los patrones o la guarda falla con los datos ahi.
EXTRA_VARS=()
if ! compgen -G "$DATA_DIR/pfd.*_p*.h5" > /dev/null && \
   compgen -G "$DATA_DIR/pfd.*.bp" > /dev/null; then
    echo "Formato detectado: ADIOS2 (.bp) -> forzando patrones"
    EXTRA_VARS=(
        "FIELD_PATTERN=$DATA_DIR/pfd.*.bp"
        "MOMENT_PATTERN=$DATA_DIR/pfd_moments.*.bp"
        "PARTICLE_PATTERN=$DATA_DIR/prt_${CASE}.*.bp"
    )
else
    echo "Formato detectado: HDF5 (.h5)"
fi

# Chequeo temprano: si no hay dumps de particulas, 'particles' y
# 'validate' van a volver a fallar igual y no vale la pena gastar el job.
if ! compgen -G "$DATA_DIR/prt_${CASE}.*" > /dev/null; then
    echo "ERROR: no hay dumps de particulas en $DATA_DIR" >&2
    echo "       Sin ellos 'particles' y 'validate' no pueden correr." >&2
    echo "       Revisar PSC_PARTICLES_EVERY del job de simulacion." >&2
    exit 1
fi

echo "============================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " DATA_DIR : $DATA_DIR"
echo " CASE     : $CASE"
echo " Etapas   : particles validate"
echo " Inicio   : $(date)"
echo "============================================"

STAGES=(particles validate)
declare -A PIDS

for stage in "${STAGES[@]}"; do
    LOG="$LOGS/analysis_${CASE}_${stage}.${SLURM_JOB_ID}.log"
    srun --nodes=1 --ntasks=1 --exclusive --job-name="$stage" \
        make "$stage" DATA_DIR="$DATA_DIR" CASE="$CASE" \
        RESULTS_ROOT="$RESULTS_ROOT" "${EXTRA_VARS[@]}" \
        > "$LOG" 2>&1 &
    PIDS[$stage]=$!
done

fail=0
for stage in "${STAGES[@]}"; do
    if wait "${PIDS[$stage]}"; then
        echo "OK: etapa '$stage' completada."
    else
        echo "ERROR: etapa '$stage' fallo. Ver $LOGS/analysis_${CASE}_${stage}.${SLURM_JOB_ID}.log" >&2
        fail=1
    fi
done

echo "============================================"
echo " Fin: $(date)"
echo "============================================"
exit "$fail"
