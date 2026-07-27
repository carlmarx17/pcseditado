#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: analysis_firehose_bimaxwellian_moderate_bigbox40
#             — COSMA7-rp-pauper
#
#  Copia de analisis_mirror_bikappa3_moderate_pauper.sh apuntando a la
#  corrida Firehose Bi-Maxwelliana moderate en caja de 40 d_i.
#
#  Corre el pipeline de analisis Python (CodeforAnalisys/Makefile,
#  target "common") sobre una corrida ya finalizada, repartiendo las 8
#  etapas independientes (brazil, fields, particles, spectral,
#  diamagnetic, heatflux, validate, physics) en 8 nodos via srun.
#
#  Envio (desde la raiz del repo en COSMA):
#    sbatch cosma_jobs/analisis/analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh
#
#  Overrides utiles (sin editar el script):
#    # otra corrida
#    sbatch --export=ALL,DATA_DIR=/ruta/a/otra_corrida  <script>
#    # mas/menos PNGs guardados y frames de GIF
#    sbatch --export=ALL,SNAPSHOT_EVERY=200000,GIF_EVERY=20000  <script>
#    # sin GIF de campos (mas rapido)
#    sbatch --export=ALL,FIELDS_GIF=  <script>
# =====================================================================

# --- Identificacion del job ---
#SBATCH --job-name=an_fh_bimax_40di

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bimax_bigbox40.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bimax_bigbox40.%J.err

# --- Particion y cuenta ---
# cosma7-rp-pauper: cola de bajo costo, limite de tiempo 24h.
# Cambiar a cosma7-rp (72h) si hace falta.
#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433

# --- Recursos ---
# 8 nodos, 1 tarea por nodo, nodo completo (28 cores) por tarea: cada
# etapa del pipeline es un proceso Python (no MPI) que puede usar
# internamente varios cores para su propio trabajo.
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# --- Tiempo maximo (maximo de la particion: 24h) ---
# La caja de 40 d_i con ngrid=1152 tiene 4x las celdas de un caso
# estandar, asi que se pide mas que las 8h de los jobs de mirror.
#SBATCH --time=16:00:00

# --- Directorio de trabajo ---
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys

# --- Notificacion por correo ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

# =====================================================================
# Entorno: mismos modulos + variables ADIOS2/HDF5 que las corridas PSC,
# para poder leer snapshots ADIOS2 (.bp) si la corrida quedo asi.
# =====================================================================
REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"

cd "$REPO/CodeforAnalisys"

RUN_ROOT=/cosma7/data/dp433/dc-mart18/anisotropy_adios2
DATA_DIR="${DATA_DIR:-$RUN_ROOT/psc_firehose_bimaxwellian_moderate_bigbox40_11643619}"
CASE=firehose_bimaxwellian_moderate_bigbox40
RESULTS_ROOT=../analysis_results

# =====================================================================
# Cadencia de figuras. SNAPSHOT_EVERY controla los PNG que se GUARDAN
# (figuras de tesis); GIF_EVERY los frames del GIF (se renderizan en
# memoria, no se guardan). Con fields_every=500 y nmax=1.2e6 la corrida
# deja ~2400 snapshots: sin este filtro el pipeline escribia ~2400 PNG
# por panel. Con 100000 quedan 13 PNG por panel (pasos 0, 100k, ..., 1.2M)
# y 121 frames de GIF.
# =====================================================================
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
GIF_EVERY="${GIF_EVERY:-10000}"
FIELDS_GIF="${FIELDS_GIF-1}"          # vaciar para no generar GIFs de campos

# Firehose es cuasi-paralela: limitar theta evita que el diagrama
# omega-k se lave con potencia oblicua en k_perp. Vaciar para desactivar.
DISPERSION_THETA_MAX="${DISPERSION_THETA_MAX-30}"

mkdir -p /cosma7/data/dp433/dc-mart18/logs

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR no existe: $DATA_DIR" >&2
    echo "       Pasa la corrida correcta con:" >&2
    echo "       sbatch --export=ALL,DATA_DIR=/ruta/a/corrida $0" >&2
    exit 1
fi

# =====================================================================
# El Makefile solo reconoce .h5 en sus guardas check-fields/moments/
# particles. Si la corrida solo dejo snapshots ADIOS2 (.bp), hay que
# forzar los patrones o esas guardas fallan aunque los datos existan.
# =====================================================================
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
