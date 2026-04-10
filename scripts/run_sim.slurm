#!/bin/bash -l
# =============================================================================
# run_sim.slurm  —  PSC Mirror Instability, un segmento del run completo
#
# DISEÑO SEGMENTADO:
#   El run de 800 000 pasos se divide en 12 segmentos de 70 000 pasos.
#   Este script corre UN segmento. El orquestador (submit_pipeline.sh)
#   encola los segmentos y las purgas con las dependencias correctas.
#
# Variables de entorno controladas por submit_pipeline.sh:
#   PSC_NMAX     : step final de este segmento (e.g. 70000, 140000, ...)
#   PSC_CHECKPOINT: path al checkpoint del segmento anterior (vacío = inicio)
#
# Uso directo (solo para depuración):
#   sbatch --export=ALL,PSC_NMAX=70000 scripts/run_sim.slurm
#   sbatch --export=ALL,PSC_NMAX=140000,PSC_CHECKPOINT=.../psc.ckpt.06.bp \
#          scripts/run_sim.slurm
#
# Para lanzar el pipeline completo:
#   bash scripts/submit_pipeline.sh
# =============================================================================
#SBATCH --job-name=mirror_seg
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --nodelist=feynman-00
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err

# ── Módulos ───────────────────────────────────────────────────────────────────
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# ── Rutas del proyecto ────────────────────────────────────────────────────────
SCRATCH="/scratchsan/observatorio/cmartinezsi/pcseditado"
EXEC="${SCRATCH}/build/src/psc_temp_aniso"
OUTPUT_DIR="${SCRATCH}/Out1"
LOG_DIR="${SCRATCH}/logs"

# Variables ADIOS2
export ADIOS2_XML_CONFIG="${SCRATCH}/adios2cfg.xml"
export OMP_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Parámetros del segmento ───────────────────────────────────────────────────
SEG_NMAX="${PSC_NMAX:-800000}"        # step final; 800000 = run completo sin segmentar
CHECKPOINT="${PSC_CHECKPOINT:-}"      # path al checkpoint anterior (vacío = inicio)

echo "========================================================"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Nodo        : $(hostname)"
echo " Ntasks      : ${SLURM_NTASKS}"
echo " PSC_NMAX    : ${SEG_NMAX}  (step final de este segmento)"
echo " Checkpoint  : ${CHECKPOINT:-<inicio fresco>}"
echo " Output dir  : ${OUTPUT_DIR}"
echo " Disco antes :"
df -h "${SCRATCH}" | tail -1
echo "========================================================"

# ── Verificar que el ejecutable existe ───────────────────────────────────────
if [ ! -f "${EXEC}" ]; then
    echo "ERROR: ejecutable no encontrado: ${EXEC}"
    echo "Compila primero con: cd build && make -j8 psc_temp_aniso"
    exit 1
fi

# ── Argumento de restart ──────────────────────────────────────────────────────
RESTART_ARG=""
if [ -n "${CHECKPOINT}" ]; then
    if [ -d "${CHECKPOINT}" ] || [ -f "${CHECKPOINT}" ]; then
        echo "Reiniciando desde checkpoint: ${CHECKPOINT}"
        RESTART_ARG="--read-checkpoint ${CHECKPOINT}"
    else
        echo "ADVERTENCIA: checkpoint no encontrado: ${CHECKPOINT}"
        echo "Iniciando en frío (sin checkpoint)."
    fi
fi

# ── Exportar PSC_NMAX para que lo lea setupParameters() ──────────────────────
export PSC_NMAX="${SEG_NMAX}"

# ── Correr simulación ─────────────────────────────────────────────────────────
cd "${OUTPUT_DIR}"

mpirun -np 72 "${EXEC}" ${RESTART_ARG}
SIM_STATUS=$?

echo ""
echo "========================================================"
echo " Segmento terminó con código: ${SIM_STATUS}"
echo " Disco tras el segmento:"
df -h "${SCRATCH}" | tail -1
du -sh "${OUTPUT_DIR}"/pfd.bp 2>/dev/null || echo "  pfd.bp: no encontrado"
echo "========================================================"

# Escribir path del checkpoint más reciente para que submit_pipeline.sh lo use
CKPT_DIR="${OUTPUT_DIR}"
LATEST_CKPT=$(ls -d "${CKPT_DIR}"/psc.ckpt.*.bp 2>/dev/null | sort -V | tail -1)
if [ -n "${LATEST_CKPT}" ]; then
    echo "${LATEST_CKPT}" > "${LOG_DIR}/latest_checkpoint_${SEG_NMAX}.txt"
    echo " Checkpoint guardado en: ${LATEST_CKPT}"
    echo " Referencia: ${LOG_DIR}/latest_checkpoint_${SEG_NMAX}.txt"
fi

exit ${SIM_STATUS}
