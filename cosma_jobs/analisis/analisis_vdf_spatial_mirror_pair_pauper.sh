#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: vdf-spatial (mirror bikappa3 <-> bimaxwellian) — COSMA7-rp-pauper
#
#  Corre el target "vdf-spatial" del Makefile (CodeforAnalisys/vdf_spatial.py)
#  sobre el UNICO par comparable de la matriz: mirror_bikappa3_moderate vs
#  mirror_bimaxwellian_moderate (mismo beta_par=5.0, A=2.0, caja, resolucion
#  y 1.2M pasos; ver src/SIMULACIONES_ANISOTROPIA.md). Este target todavia
#  no se habia corrido para ninguno de los dos casos: "common" (make analysis)
#  no lo incluye, hay que pedirlo explicitamente.
#
#  Que hace vdf_spatial.py: clasifica particulas por el |B| local de su celda
#  (percentil bajo = hueco magnetico, alto = pico, resto = ambiente) y compara
#  f(v_par), f(v_perp) entre poblaciones -- la pregunta central del nicho de
#  tesis (VDF dentro de magnetic holes, bi-kappa vs bi-Maxwelliana). Tambien
#  produce el mapa de anisotropia por macro-celda.
#
#  Prerrequisito: necesita dumps de particulas (prt_<CASE>.*) en DATA_DIR.
#  mirror_bikappa3_moderate ya fallo una vez en las etapas 'particles' y
#  'validate' (ver analisis_mirror_bikappa3_moderate_rerun_particles.sh); si
#  ese rerun tampoco dejo prt_mirror_bikappa3_moderate.* en el DATA_DIR de
#  abajo, este job va a fallar en el precheck de esa rama con el mismo motivo
#  (revisar PSC_PARTICLES_EVERY del job de simulacion). El lado bimaxwellian
#  si tiene 03_particles en analysis_results/, o sea que sus dumps existen.
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/analisis/analisis_vdf_spatial_mirror_pair_pauper.sh
# =====================================================================

# --- Identificacion del job ---
#SBATCH --job-name=an_vdf_spatial

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_vdf_spatial_mirror_pair.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_vdf_spatial_mirror_pair.%J.err

# --- Particion y cuenta ---
#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433

# --- Recursos: 2 nodos, uno por caso, en paralelo ---
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

RESULTS_ROOT=../analysis_results
LOGS=/cosma7/data/dp433/dc-mart18/logs
mkdir -p "$LOGS"

# Un caso por indice: DATA_DIR y CASE deben quedar en el mismo orden.
DATA_DIRS=(
    /cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_bikappa3_moderate_11618877
    /cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_bimaxwellian_moderate_11596993
)
CASES=(
    mirror_bikappa3_moderate
    mirror_bimaxwellian_moderate
)

echo "============================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Casos    : ${CASES[*]}"
echo " Inicio   : $(date)"
echo "============================================"

declare -A PIDS

for i in "${!CASES[@]}"; do
    DATA_DIR="${DATA_DIRS[$i]}"
    CASE="${CASES[$i]}"
    LOG="$LOGS/analysis_${CASE}_vdf_spatial.${SLURM_JOB_ID}.log"

    # El Makefile solo reconoce .h5 en sus guardas; si la corrida quedo en
    # ADIOS2 hay que forzar los patrones o la guarda falla con los datos ahi.
    EXTRA_VARS=()
    if ! compgen -G "$DATA_DIR/pfd.*_p*.h5" > /dev/null && \
       compgen -G "$DATA_DIR/pfd.*.bp" > /dev/null; then
        EXTRA_VARS=(
            "FIELD_PATTERN=$DATA_DIR/pfd.*.bp"
            "MOMENT_PATTERN=$DATA_DIR/pfd_moments.*.bp"
            "PARTICLE_PATTERN=$DATA_DIR/prt_${CASE}.*.bp"
        )
    fi

    # Chequeo temprano: sin dumps de particulas, vdf-spatial no puede correr.
    if ! compgen -G "$DATA_DIR/prt_${CASE}.*" > /dev/null; then
        {
            echo "ERROR: no hay dumps de particulas en $DATA_DIR"
            echo "       Sin ellos 'vdf-spatial' no puede correr."
            echo "       Revisar PSC_PARTICLES_EVERY del job de simulacion."
        } > "$LOG" 2>&1
        echo "ERROR: caso '$CASE' sin dumps de particulas (ver $LOG)." >&2
        PIDS[$CASE]=""
        continue
    fi

    srun --nodes=1 --ntasks=1 --exclusive --job-name="vdf_$CASE" \
        make vdf-spatial DATA_DIR="$DATA_DIR" CASE="$CASE" \
        RESULTS_ROOT="$RESULTS_ROOT" "${EXTRA_VARS[@]}" \
        > "$LOG" 2>&1 &
    PIDS[$CASE]=$!
done

fail=0
for CASE in "${CASES[@]}"; do
    pid="${PIDS[$CASE]}"
    if [ -z "$pid" ]; then
        fail=1
        continue
    fi
    if wait "$pid"; then
        echo "OK: '$CASE' completado."
    else
        echo "ERROR: '$CASE' fallo. Ver $LOGS/analysis_${CASE}_vdf_spatial.${SLURM_JOB_ID}.log" >&2
        fail=1
    fi
done

echo "============================================"
echo " Fin: $(date)"
echo "============================================"
exit "$fail"
