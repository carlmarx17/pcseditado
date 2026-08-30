#!/bin/bash
#
# =====================================================================
#  merge_run_folders.sh — inspecciona y (si procede) une en una sola
#  carpeta las corridas de un mismo caso que quedaron partidas.
#
#  POR QUE EXISTE
#  --------------
#  sim_firehose_bikappa3_40di.sh define
#      RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"
#  y nunca exporta PSC_RESTART. psc_anisotropy_case.hxx solo reanuda
#  desde checkpoint si PSC_RESTART esta definido (ver el getenv en la
#  linea ~449). Consecuencia: cada reenvio creo una carpeta NUEVA y
#  arranco desde t=0 otra vez.
#
#  Por eso las carpetas _11657054 y _11657093 NO son tramos
#  consecutivos de una misma corrida: son corridas independientes que
#  probablemente cubren el MISMO rango de pasos. Si es asi, unirlas no
#  reconstruye nada — mezclaria dos trayectorias distintas en una sola
#  serie temporal, que es peor que quedarse con la mas larga.
#
#  Este script primero MIDE (modo por defecto, no toca nada) y solo
#  une si los rangos son disjuntos y se pasa --merge explicitamente.
#
#  USO
#  ---
#    # 1) diagnostico, no escribe nada:
#    ./merge_run_folders.sh psc_firehose_bikappa3_bigbox40
#
#    # 2) si y solo si el diagnostico dice que son disjuntas:
#    ./merge_run_folders.sh --merge psc_firehose_bikappa3_bigbox40
#
#  La union usa hardlinks: misma particion, cero espacio extra, y las
#  carpetas originales quedan intactas.
# =====================================================================

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/cosma7/data/dp433/dc-mart18/anisotropy_adios2}"
DO_MERGE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --merge) DO_MERGE=1; shift ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "uso: $0 [--merge] [--run-root DIR] <prefijo-del-target>" >&2
    echo "ej:  $0 psc_firehose_bikappa3_bigbox40" >&2
    exit 2
fi

TARGET="$1"
MERGED_DIR="$RUN_ROOT/${TARGET}_merged"

# ---------------------------------------------------------------------
# Extrae el numero de paso de un nombre de archivo de salida de PSC:
#   pfd.001200000_p0000.h5      -> 1200000
#   pfd_moments.000500000_p0.h5 -> 500000
#   prt_<basename>.000010000.h5 -> 10000
#   checkpoint_000150000.bp     -> 150000
# ---------------------------------------------------------------------
steps_of() {
    local dir="$1" pattern="$2"
    find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null \
        | sed -E 's|.*/||; s|_p[0-9]+\.h5$||; s|\.h5$||; s|\.bp$||; s|^[^.]*\.||; s|^checkpoint_||' \
        | grep -E '^[0-9]+$' \
        | sed 's/^0*//; s/^$/0/' \
        | sort -n -u
}

report_dir() {
    local d="$1"
    local name; name="$(basename "$d")"
    local pfd_steps; pfd_steps="$(steps_of "$d" 'pfd.*')"
    local n; n="$(printf '%s' "$pfd_steps" | grep -c . || true)"

    if [ "$n" -eq 0 ]; then
        printf '  %-52s  sin snapshots de campos\n' "$name"
        return
    fi

    local first last
    first="$(printf '%s\n' "$pfd_steps" | head -1)"
    last="$(printf '%s\n' "$pfd_steps"  | tail -1)"

    local nprt nckpt
    nprt="$(find "$d" -maxdepth 1 -name 'prt_*.h5' 2>/dev/null | wc -l)"
    nckpt="$(find "$d" -maxdepth 1 -name 'checkpoint_*.bp' 2>/dev/null | wc -l)"

    printf '  %-52s  %5d pasos  [%d .. %d]  prt=%d  ckpt=%d\n' \
        "$name" "$n" "$first" "$last" "$nprt" "$nckpt"
}

# ---------------------------------------------------------------------
# Diagnostico
# ---------------------------------------------------------------------
mapfile -t DIRS < <(find "$RUN_ROOT" -maxdepth 1 -type d -name "${TARGET}_*" \
                      ! -name '*_merged' | sort)

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no hay carpetas ${TARGET}_* en $RUN_ROOT" >&2
    exit 1
fi

echo "== Carpetas de $TARGET en $RUN_ROOT =="
for d in "${DIRS[@]}"; do report_dir "$d"; done
echo

# Solapamiento: cuantos pasos aparecen en mas de una carpeta.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for d in "${DIRS[@]}"; do
    steps_of "$d" 'pfd.*' > "$tmp/$(basename "$d").steps"
done

total_unique="$(cat "$tmp"/*.steps | sort -n -u | wc -l)"
total_sum="$(cat "$tmp"/*.steps | wc -l)"
overlap=$(( total_sum - total_unique ))

echo "pasos distintos en total : $total_unique"
echo "suma de pasos por carpeta: $total_sum"
echo "pasos duplicados         : $overlap"
echo

if [ "$overlap" -gt 0 ]; then
    echo "VEREDICTO: las carpetas SE SOLAPAN en $overlap pasos."
    echo
    echo "  Son corridas independientes desde t=0, no tramos de una misma"
    echo "  corrida. Unirlas mezclaria dos trayectorias distintas en una"
    echo "  sola serie temporal: los saltos en t no serian fisica."
    echo
    echo "  Lo correcto es quedarse con la carpeta mas larga y, si no"
    echo "  llega a saturacion, REANUDARLA desde su ultimo checkpoint:"
    echo
    echo "    sbatch --export=ALL,RUN_TAG=<tag>,PSC_RESTART=<checkpoint>.bp \\"
    echo "      cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh"
    echo
    if [ "$DO_MERGE" -eq 1 ]; then
        echo "--merge IGNORADO: no se une algo que se solapa." >&2
        exit 3
    fi
    exit 0
fi

echo "VEREDICTO: los rangos son DISJUNTOS — se pueden unir sin ambiguedad."

if [ "$DO_MERGE" -eq 0 ]; then
    echo
    echo "Diagnostico solamente. Para unir de verdad:"
    echo "  $0 --merge $TARGET"
    exit 0
fi

# ---------------------------------------------------------------------
# Union por hardlinks. No borra ni mueve nada de las carpetas fuente.
# ---------------------------------------------------------------------
echo
echo "== Uniendo en $MERGED_DIR =="
mkdir -p "$MERGED_DIR"

linked=0
skipped=0
for d in "${DIRS[@]}"; do
    while IFS= read -r f; do
        base="$(basename "$f")"
        if [ -e "$MERGED_DIR/$base" ]; then
            skipped=$(( skipped + 1 ))
            continue
        fi
        if ! ln "$f" "$MERGED_DIR/$base" 2>/dev/null; then
            cp -a "$f" "$MERGED_DIR/$base"   # particiones distintas
        fi
        linked=$(( linked + 1 ))
    done < <(find "$d" -maxdepth 1 -type f \
                \( -name 'pfd.*' -o -name 'pfd_moments.*' -o -name 'prt_*.h5' \))
done

# adios2cfg.xml y el ejecutable, para que la carpeta sea autocontenida.
for extra in adios2cfg.xml "$TARGET"; do
    [ -e "$MERGED_DIR/$extra" ] && continue
    for d in "${DIRS[@]}"; do
        [ -e "$d/$extra" ] && { cp -a "$d/$extra" "$MERGED_DIR/"; break; }
    done
done

echo "archivos enlazados: $linked"
echo "ya presentes      : $skipped"
echo
echo "Carpeta unida: $MERGED_DIR"
echo "Las originales quedaron intactas."
echo
echo "Analiza con:"
echo "  sbatch --export=ALL,DATA_DIR=$MERGED_DIR \\"
echo "    cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh"
