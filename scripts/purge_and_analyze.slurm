#!/bin/bash -l
# =============================================================================
# purge_and_analyze.slurm  —  Análisis y limpieza de UN segmento de simulación
#
# DISEÑO SEGMENTADO:
#   Corre después de cada segmento de simulación con dependency=after:<SIM_JOB>
#   (dependency=after, no afterok → corre incluso si la sim falla parcialmente)
#
#   Acciones:
#     1. Analiza campos BP5 (pfd.bp) → HDF5 compacto (APPEND al existente)
#     2. Analiza momentos si existen (pfd_moments.bp)
#     3. Borra pfd.bp, pfd_moments.bp ya procesados
#     4. Borra checkpoints anteriores al más reciente
#     5. Borra archivos prt*.bp de este segmento (submuestra → HDF5 primero)
#
# Variables de entorno (puestas por submit_pipeline.sh):
#   PSC_SEG_END : step final del segmento que acaba de terminar (para logs)
#   PSC_NMAX    : mismo valor (para identificación)
#
# Se puede lanzar manualmente:
#   sbatch --export=ALL,PSC_SEG_END=70000 scripts/purge_and_analyze.slurm
# =============================================================================
#SBATCH --job-name=purge_seg
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --nodelist=feynman-00
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/purge_%j.out
#SBATCH --error=logs/purge_%j.err

# ── Módulos ───────────────────────────────────────────────────────────────────
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2
# module load Python/3.11  # descomenta si el cluster lo requiere

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRATCH="/scratchsan/observatorio/cmartinezsi/pcseditado"
OUTPUT_DIR="${SCRATCH}/Out1"
ANALYSIS_DIR="${SCRATCH}/CodeforAnalisys"
DIAG_DIR="${SCRATCH}/diagnostics_compact"
LOG_DIR="${SCRATCH}/logs"
SEG_END="${PSC_SEG_END:-desconocido}"

mkdir -p "${DIAG_DIR}" "${LOG_DIR}"

echo "========================================================"
echo " Job ID     : ${SLURM_JOB_ID}"
echo " Nodo       : $(hostname)"
echo " Segmento   : hasta step ${SEG_END}"
echo " Output dir : ${OUTPUT_DIR}"
echo " Diag dir   : ${DIAG_DIR}"
echo " Disco ANTES de purga:"
df -h "${SCRATCH}" | tail -1
du -sh "${OUTPUT_DIR}" 2>/dev/null | head -5
echo "========================================================"

# =============================================================================
# 1. Analizar campos instantáneos (pfd.bp) → HDF5 compacto
#    streaming_analysis.py usa mode="a" → APPEND al HDF5 existente
# =============================================================================
echo ""
echo "[1/5] Analizando campos EM (pfd.bp)..."
PFD_BP="${OUTPUT_DIR}/pfd.bp"
if [ -d "${PFD_BP}" ]; then
    python3 "${ANALYSIS_DIR}/streaming_analysis.py" \
        --mode fields \
        --input-dir  "${OUTPUT_DIR}" \
        --output-dir "${DIAG_DIR}" \
        --n-workers  8
    FIELDS_STATUS=$?
    if [ ${FIELDS_STATUS} -ne 0 ]; then
        echo "  ERROR: análisis de pfd.bp falló (código ${FIELDS_STATUS})."
        echo "  ABORT: no se borrarán datos de este segmento."
        echo "  Revisar ${LOG_DIR}/purge_${SLURM_JOB_ID}.err para detalles."
        exit 1
    fi
    echo "  OK: pfd.bp procesado exitosamente."
else
    echo "  SKIP: pfd.bp no encontrado en ${OUTPUT_DIR}."
fi

# =============================================================================
# 2. Analizar momentos si existen (pfd_moments.bp)
# =============================================================================
echo ""
echo "[2/5] Analizando momentos (pfd_moments.bp)..."
MOMENTS_BP="${OUTPUT_DIR}/pfd_moments.bp"
if [ -d "${MOMENTS_BP}" ]; then
    python3 "${ANALYSIS_DIR}/streaming_analysis.py" \
        --mode fields \
        --input-dir  "${OUTPUT_DIR}" \
        --output-dir "${DIAG_DIR}" \
        --n-workers  4
    echo "  OK: momentos procesados."
else
    echo "  SKIP: pfd_moments.bp no existe (intervalo=4000, puede no haber en este seg)."
fi

# =============================================================================
# 3. Borrar SOLO los BP5 ya procesados (nunca borrar checkpoint)
# =============================================================================
echo ""
echo "[3/5] Borrando BP5 de campos ya analizados..."
for bp_dir in "${OUTPUT_DIR}/pfd.bp" \
              "${OUTPUT_DIR}/pfd_moments.bp" \
              "${OUTPUT_DIR}/tfd.bp" \
              "${OUTPUT_DIR}/tfd_moments.bp"; do
    if [ -d "${bp_dir}" ]; then
        SIZE=$(du -sh "${bp_dir}" 2>/dev/null | cut -f1)
        echo "  Borrando [${SIZE}]: ${bp_dir}"
        rm -rf "${bp_dir}"
        echo "  OK: borrado."
    fi
done

# =============================================================================
# 4. Particulas: submuestra 0.5% → HDF5, luego borra prt*.bp
# =============================================================================
echo ""
echo "[4/5] Submuestreando y purgando partículas..."
for prt_dir in "${OUTPUT_DIR}"/prt*.bp; do
    if [ -d "${prt_dir}" ]; then
        echo "  Procesando: ${prt_dir}"
        python3 "${ANALYSIS_DIR}/streaming_analysis.py" \
            --mode particles \
            --input-dir  "${prt_dir}" \
            --output-dir "${DIAG_DIR}" \
            --sample-fraction 0.005
        if [ $? -eq 0 ]; then
            SIZE=$(du -sh "${prt_dir}" 2>/dev/null | cut -f1)
            echo "  Borrando partículas [${SIZE}]: ${prt_dir}"
            rm -rf "${prt_dir}"
        else
            echo "  WARNING: submuestra de ${prt_dir} falló — BP5 conservado."
        fi
    fi
done

# =============================================================================
# 5. Conservar solo el checkpoint más reciente
#    PSC escribe psc.ckpt.NNN.bp con N creciente. Solo necesitamos el último.
# =============================================================================
echo ""
echo "[5/5] Podando checkpoints antiguos..."
# Busca todos los checkpoint dirs en Output_DIR
mapfile -t CKPTS < <(ls -d "${OUTPUT_DIR}"/psc.ckpt.*.bp 2>/dev/null | sort -V)
N=${#CKPTS[@]}
if [ "${N}" -gt 1 ]; then
    for (( i=0; i<N-1; i++ )); do
        SIZE=$(du -sh "${CKPTS[$i]}" 2>/dev/null | cut -f1)
        echo "  Borrando checkpoint viejo [${SIZE}]: ${CKPTS[$i]}"
        rm -rf "${CKPTS[$i]}"
    done
    echo "  Conservado (más reciente): ${CKPTS[$((N-1))]}"
elif [ "${N}" -eq 1 ]; then
    echo "  Solo 1 checkpoint — conservado: ${CKPTS[0]}"
else
    echo "  Sin checkpoints encontrados en ${OUTPUT_DIR}."
fi

# =============================================================================
# Resumen final
# =============================================================================
echo ""
echo "========================================================"
echo " Disco DESPUÉS de purga del segmento hasta step ${SEG_END}:"
df -h "${SCRATCH}" | tail -1
echo " Contenido de ${OUTPUT_DIR}:"
du -sh "${OUTPUT_DIR}"/* 2>/dev/null || echo "  (vacío o checkpoint solamente)"
echo " HDF5 diagnósticos acumulados:"
du -sh "${DIAG_DIR}"/* 2>/dev/null || echo "  (ninguno)"
echo " Purga completada. OK."
echo "========================================================"
