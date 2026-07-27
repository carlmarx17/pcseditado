#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: analysis_firehose_bikappa3_bigbox40 — COSMA7-rp-pauper
#
#  Gemelo de analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh
#  para la corrida Bi-Kappa (kappa=3) en caja de 40 d_i.
#
#  OJO: hay TRES carpetas de esta corrida en anisotropy_adios2
#  (11654252, 11657054, 11657093), una por cada envio del job de
#  simulacion. sim_firehose_bikappa3_40di.sh crea un RUN_DIR nuevo por
#  SLURM_JOB_ID, asi que los snapshots quedan REPARTIDOS y ninguna
#  carpeta contiene la corrida completa. Este script, si no le pasas
#  DATA_DIR, elige la carpeta con MAS snapshots de campos y avisa de las
#  otras. Antes de usar el resultado como definitivo, confirma cual de
#  las tres es la buena (ver .out/.err de cada job id).
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
#
#  Overrides utiles (sin editar el script):
#    sbatch --export=ALL,DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_firehose_bikappa3_bigbox40_11657093  <script>
#    sbatch --export=ALL,SNAPSHOT_EVERY=200000,GIF_EVERY=20000  <script>
#    sbatch --export=ALL,FIELDS_GIF=  <script>
# =====================================================================

# --- Identificacion del job ---
#SBATCH --job-name=an_fh_bik3_40di

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bikappa3_bigbox40.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bikappa3_bigbox40.%J.err

# --- Particion y cuenta ---
#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433

# --- Recursos ---
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# --- Tiempo maximo (maximo de la particion: 24h) ---
#SBATCH --time=16:00:00

# --- Directorio de trabajo ---
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys

# --- Notificacion por correo ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

cd "$REPO/CodeforAnalisys"

RUN_ROOT=/cosma7/data/dp433/dc-mart18/anisotropy_adios2
CASE=firehose_bikappa3_bigbox40
RESULTS_ROOT=../analysis_results

# =====================================================================
# Seleccion de carpeta: si no se paso DATA_DIR, elegir entre las
# psc_firehose_bikappa3_bigbox40_<JOBID> la que tenga mas snapshots de
# campos, y listar todas para que quede en el log cual se descarto.
# =====================================================================
count_snapshots() {
    local d="$1"
    find "$d" -maxdepth 1 \( -name 'pfd.*.bp' -o -name 'pfd.*_p*.h5' \) 2>/dev/null | wc -l
}

if [ -z "${DATA_DIR:-}" ]; then
    best_dir=""
    best_n=-1
    echo "Carpetas candidatas de $CASE:"
    for d in "$RUN_ROOT"/psc_firehose_bikappa3_bigbox40_*; do
        [ -d "$d" ] || continue
        n=$(count_snapshots "$d")
        echo "  $(basename "$d"): $n snapshots de campos"
        if [ "$n" -gt "$best_n" ]; then
            best_n="$n"
            best_dir="$d"
        fi
    done
    if [ -z "$best_dir" ]; then
        echo "ERROR: no se encontro ninguna carpeta psc_firehose_bikappa3_bigbox40_* en $RUN_ROOT" >&2
        exit 1
    fi
    DATA_DIR="$best_dir"
    echo "Elegida (mas snapshots): $DATA_DIR"
    echo "Si no es la corrida buena, reenvia con --export=ALL,DATA_DIR=..."
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR no existe: $DATA_DIR" >&2
    exit 1
fi

# =====================================================================
# Cadencia de figuras: SNAPSHOT_EVERY = PNG guardados (figuras de
# tesis), GIF_EVERY = frames del GIF (renderizados en memoria).
# Con fields_every=500 y nmax=1.2e6 hay ~2400 snapshots; 100000 deja
# 13 PNG por panel y 121 frames de GIF.
# =====================================================================
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
GIF_EVERY="${GIF_EVERY:-10000}"
FIELDS_GIF="${FIELDS_GIF-1}"
DISPERSION_THETA_MAX="${DISPERSION_THETA_MAX-30}"

mkdir -p /cosma7/data/dp433/dc-mart18/logs

MAKE_VARS=(
    "SNAPSHOT_EVERY=$SNAPSHOT_EVERY"
    "GIF_EVERY=$GIF_EVERY"
    "FIELDS_GIF=$FIELDS_GIF"
    "DISPERSION_THETA_MAX=$DISPERSION_THETA_MAX"
)
if ! compgen -G "$DATA_DIR/pfd.*_p*.h5" > /dev/null && \
   compgen -G "$DATA_DIR/pfd.*.bp" > /dev/null; then
    echo "Formato detectado: ADIOS2 (.bp) -> forzando FIELD/MOMENT/PARTICLE_PATTERN"
    MAKE_VARS+=(
        "FIELD_PATTERN=$DATA_DIR/pfd.*.bp"
        "MOMENT_PATTERN=$DATA_DIR/pfd_moments.*.bp"
        "PARTICLE_PATTERN=$DATA_DIR/prt_${CASE}.*.bp"
    )
else
    echo "Formato detectado: HDF5 (.h5)"
fi

echo "============================================"
echo " Job ID          : $SLURM_JOB_ID"
echo " Nodos           : $SLURM_JOB_NODELIST"
echo " DATA_DIR        : $DATA_DIR"
echo " CASE            : $CASE"
echo " SNAPSHOT_EVERY  : $SNAPSHOT_EVERY  (PNG guardados)"
echo " GIF_EVERY       : $GIF_EVERY  (frames de GIF)"
echo " Inicio          : $(date)"
echo "============================================"

# =====================================================================
# 1) manifest: rapido, corre primero en un solo nodo de la asignacion
# =====================================================================
srun --nodes=1 --ntasks=1 --exclusive \
    make manifest DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${MAKE_VARS[@]}"
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
        make "$stage" DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${MAKE_VARS[@]}" \
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
