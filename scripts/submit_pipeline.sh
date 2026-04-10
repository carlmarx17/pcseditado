#!/bin/bash
# =============================================================================
# submit_pipeline.sh — Orquestador del pipeline segmentado de simulación PSC
#
# FUNCIONAMIENTO:
#   Divide el run de 800 000 pasos en N_SEGS segmentos de N_SEGMENT pasos.
#   Por cada segmento encola:
#     - run_sim.slurm          → corre PSC hasta el step límite del segmento
#     - purge_and_analyze.slurm → analiza BP5 y los borra
#
#   Dependencias SLURM:
#     purge[k]   --dependency=after:sim[k]     (purga corre tras sim del segmento)
#     sim[k+1]   --dependency=after:purge[k]   (sig. sim espera purga anterior)
#
#   Nota: se usa "after" (no "afterok") para que purge corra incluso si la
#   sim termina con error. El purge.slurm aborta si el análisis falla.
#
# GARANTÍA DE DISCO:
#   Cada segmento produce máx. 70 snaps × 90 MB ZFP = ~6.3 GB.
#   La purga borra BP5 antes de que empiece el siguiente segmento.
#   → Pico en disco: ~6.3 GB datos + ~4 GB checkpoint = ~10.3 GB.
#   Para ser más conservador usar N_SEGMENT=55000 → ~5 GB datos.
#
# USO:
#   bash scripts/submit_pipeline.sh                  # inicio fresco
#   bash scripts/submit_pipeline.sh --dry-run        # solo imprime jobs
#   bash scripts/submit_pipeline.sh --resume 210000  # reanuda desde step
#
# MONITOREO:
#   squeue -M fisica -u cmartinezsi
#   watch -n 30 'squeue -M fisica -u cmartinezsi'
#   watch -n 60 'df -h /scratchsan; du -sh /scratchsan/.../Out1/'
# =============================================================================

set -euo pipefail

# ── Configuración ─────────────────────────────────────────────────────────────
SCRATCH="/scratchsan/observatorio/cmartinezsi/pcseditado"
SCRIPTS_DIR="${SCRATCH}/scripts"
LOG_DIR="${SCRATCH}/logs"

N_TOTAL=800000       # pasos totales de la simulación
N_SEGMENT=70000      # pasos por segmento → 70 snaps × 90 MB ≈ 6.3 GB < 7 GB
                     # usa 55000 si el cupo es estrictamente 7 GB

CLUSTERS="fisica"
PARTITION="cpu.cecc"
NODE="feynman-00"

# ── Flags de CLI ──────────────────────────────────────────────────────────────
DRY_RUN=0
RESUME_FROM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;            shift ;;
        --resume)   RESUME_FROM="${2:-0}"; shift 2 ;;
        *)          echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

# ── Función auxiliar: sbatch o echo en dry-run ────────────────────────────────
submit() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "99999"   # ID ficticio para que el script pueda continuar
    else
        sbatch --parsable "$@"
    fi
}

# ── Calcular segmentos ────────────────────────────────────────────────────────
N_SEGS=$(( (N_TOTAL + N_SEGMENT - 1) / N_SEGMENT ))  # ceil

echo "========================================================"
echo "  PSC Mirror Instability — Pipeline segmentado"
echo "  Total steps  : ${N_TOTAL}"
echo "  Segmentos    : ${N_SEGS}  (${N_SEGMENT} pasos c/u)"
echo "  Nodo         : ${NODE}  (72 cores)"
echo "  Dry-run      : ${DRY_RUN}"
[ "${RESUME_FROM}" -gt 0 ] && echo "  Reanudando desde step: ${RESUME_FROM}"
echo "========================================================"
echo ""

# ── Variables de estado ───────────────────────────────────────────────────────
PREV_PURGE_JOB=""   # ID del job de purga anterior (dependency del siguiente sim)
ALL_JOBS=()         # lista de todos los job IDs para cancelación masiva

# ── Loop de segmentos ─────────────────────────────────────────────────────────
for seg in $(seq 1 "${N_SEGS}"); do

    SEG_START=$(( (seg - 1) * N_SEGMENT ))
    SEG_END=$(( seg * N_SEGMENT ))
    [ "${SEG_END}" -gt "${N_TOTAL}" ] && SEG_END="${N_TOTAL}"

    # Saltar segmentos ya completados si estamos reanudando
    if [ "${RESUME_FROM}" -gt 0 ] && [ "${SEG_END}" -le "${RESUME_FROM}" ]; then
        echo "  Seg ${seg} (${SEG_START}→${SEG_END}): OMITIDO (--resume ${RESUME_FROM})"
        continue
    fi

    echo "  ── Segmento ${seg}/${N_SEGS}: steps ${SEG_START} → ${SEG_END} ──"

    # ── Dependency del sim: esperar purga del segmento anterior ──────────────
    SIM_DEP_ARGS=()
    if [ -n "${PREV_PURGE_JOB}" ]; then
        SIM_DEP_ARGS=(--dependency="after:${PREV_PURGE_JOB}")
    fi

    # ── Job de simulación ─────────────────────────────────────────────────────
    SIM_JOB=$(submit \
        --job-name="mirror_s${seg}" \
        --clusters="${CLUSTERS}" \
        --partition="${PARTITION}" \
        --nodelist="${NODE}" \
        --qos=low \
        --nodes=1 \
        --ntasks=72 \
        --exclusive \
        --time=2-00:00:00 \
        --output="${LOG_DIR}/sim_s${seg}_%j.out" \
        --error="${LOG_DIR}/sim_s${seg}_%j.err" \
        --export="ALL,PSC_NMAX=${SEG_END},PSC_SEG_END=${SEG_END}" \
        "${SIM_DEP_ARGS[@]+"${SIM_DEP_ARGS[@]}"}" \
        "${SCRIPTS_DIR}/run_sim.slurm"
    )
    echo "    Sim  job: ${SIM_JOB}  [PSC_NMAX=${SEG_END}]"
    ALL_JOBS+=("${SIM_JOB}")

    # ── Job de purga: corre DESPUÉS del sim (con after, no afterok) ───────────
    PURGE_JOB=$(submit \
        --job-name="purge_s${seg}" \
        --clusters="${CLUSTERS}" \
        --partition="${PARTITION}" \
        --nodelist="${NODE}" \
        --qos=low \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --time=03:00:00 \
        --output="${LOG_DIR}/purge_s${seg}_%j.out" \
        --error="${LOG_DIR}/purge_s${seg}_%j.err" \
        --export="ALL,PSC_SEG_END=${SEG_END}" \
        --dependency="after:${SIM_JOB}" \
        "${SCRIPTS_DIR}/purge_and_analyze.slurm"
    )
    echo "    Purge job: ${PURGE_JOB}  [dependency=after:${SIM_JOB}]"
    ALL_JOBS+=("${PURGE_JOB}")

    PREV_PURGE_JOB="${PURGE_JOB}"
    echo ""
done

# ── Resumen ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Jobs encolados: ${#ALL_JOBS[@]}  (${N_SEGS} sim + ${N_SEGS} purge)"
echo "  IDs            : ${ALL_JOBS[*]}"
echo ""
echo "  Monitoreo:"
echo "    squeue -M ${CLUSTERS} -u \$USER"
echo "    watch -n 30 'squeue -M ${CLUSTERS} -u \$USER'"
echo ""
echo "  Uso de disco en tiempo real:"
echo "    watch -n 60 'df -h /scratchsan; du -sh ${SCRATCH}/Out1/'"
echo ""
echo "  Ver logs del segmento actual:"
echo "    tail -f ${LOG_DIR}/sim_s*_*.out | head -50"
echo "    tail -f ${LOG_DIR}/purge_s*_*.out | head -50"
echo ""
echo "  Cancelar TODO el pipeline:"
echo "    scancel -M ${CLUSTERS} ${ALL_JOBS[*]}"
echo ""
echo "  Reanudar desde un step (si falla un segmento):"
echo "    bash ${SCRIPTS_DIR}/submit_pipeline.sh --resume <STEP_COMPLETADO>"
echo "========================================================"
