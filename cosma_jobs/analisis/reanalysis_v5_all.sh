#!/bin/bash -l
#
# =====================================================================
#  reanalysis_v5_all.sh — clean the old analysis products and re-run the
#  full v5 pipeline (analysis conventions version 5, 2026-09-28) on every
#  FINISHED anisotropy run, spread over up to 20 COSMA7 nodes.
#
#  Why: the v5 revision changes numbers that go into the thesis (gamma of
#  transverse modes was 2x, gamma(k) biased low, J_dia sign, heat flux was
#  a proxy, electron-scale cut for whistler). Every product made with the
#  old code must be replaced; see CodeforAnalisys/AUDITORIA_FISICA_PAPER.md,
#  "Follow-up (2026-09-28)".
#
#  Runs analysed (anisotropy_adios2/):
#    firehose_bimaxwellian_strong_bigbox40    psc_..._12062436  (the other
#      folder of this case, _12031167, is not used)
#    firehose_bikappa3_bigbox40               psc_..._11657093
#    mirror_bimaxwellian_moderate             psc_..._11596993
#    mirror_bikappa3_moderate                 psc_..._11618877
#    mirror_bikappa5_moderate                 psc_..._12063822
#  Left out on purpose, not finished yet (their old results are kept):
#    firehose_bimaxwellian_moderate_bigbox40  _11643619
#    whistler_bimaxwellian_strong_mr800       _12068623 (mi/me = 800: it
#      will also need its own profile in psc_units.py before analysis)
#  The job re-checks completeness itself: a run whose last field snapshot
#  is short of the profile's nmax is skipped (ALLOW_INCOMPLETE=1 forces it).
#
#  Order of work (nothing is deleted until the new code has proven itself):
#    0. the COSMA checkout must contain the v5 pipeline (git pull first);
#    1. preflight: unit tests + synthetic end-to-end run on a compute node;
#    2. completeness check of every run;
#    3. per run: new manifest + initial-condition check; ONLY if it passes,
#       the run's old results are deleted (run_aware_v4/<case>,
#       analysis_results/<case>) together with the old comparisons;
#    4. every analysis stage of every run as its own job step, longest
#       first, at most one step per node (one Python process per stage;
#       the heavy ones use all 28 cores through multiprocessing);
#    5. comparisons of the controlled series (same beta, A, grid, box, ppc;
#       only the distribution changes), compare-physics + kappa_evolution:
#         mirror moderate 20 d_i:  bi-Maxwellian / kappa 5 / kappa 3
#         firehose strong 40 d_i:  bi-Maxwellian / kappa 3
#       (both firehose job scripts use identical numerical settings).
#    6. summary: <NEW_ROOT>/REANALYSIS_SUMMARY_<jobid>.txt
#
#  Submit (from the repository root on COSMA, after `git pull`):
#    sbatch cosma_jobs/analisis/reanalysis_v5_all.sh
#
#  Useful overrides (no editing needed):
#    sbatch --export=ALL,ONLY="mirror_bikappa5_moderate" <script>   # subset
#    sbatch --export=ALL,CLEAN=0 <script>          # keep the old results
#    sbatch --export=ALL,DRY_RUN=1 <script>        # print, run nothing
#    sbatch --export=ALL,GROWTH_T_START=5,GROWTH_T_END=20 <script>
#    sbatch --nodes=12 --ntasks=12 <script>        # fewer nodes, same result
# =====================================================================

# --- Job identification ---
#SBATCH --job-name=reanalysis_v5

# --- Output ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/reanalysis_v5.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/reanalysis_v5.%J.err

# --- Partition and account ---
# cosma7-rp: same node pool as cosma7-rp-pauper, 72 h limit instead of 24 h.
# The two 40 d_i runs (1152^2, ~2400 snapshots) need more than 24 h for the
# heaviest stages.
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433

# --- Resources: 20 full nodes, one analysis stage per node at a time ---
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# --- Wall time ---
#SBATCH --time=48:00:00

# --- Working directory ---
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys

# --- Mail ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"
cd "$REPO/CodeforAnalisys" || exit 1

RUN_ROOT=/cosma7/data/dp433/dc-mart18/anisotropy_adios2
RESULTS_BASE="$REPO/analysis_results"
NEW_ROOT="${NEW_ROOT:-$RESULTS_BASE/v5}"
OLD_ROOTS=("$RESULTS_BASE/run_aware_v4" "$RESULTS_BASE")
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="/cosma7/data/dp433/dc-mart18/logs/reanalysis_v5_${JOB_ID}"
STATUS_DIR="$LOG_DIR/status"
PY="${PYTHON:-$REPO/.venv/bin/python}"

CLEAN="${CLEAN:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
ONLY="${ONLY:-}"
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_JOB_NUM_NODES:-20}}"
POLL="${POLL:-20}"       # seconds between checks when `wait -n` is unavailable
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
GIF_EVERY="${GIF_EVERY:-10000}"
FIELDS_GIF="${FIELDS_GIF-1}"
GROWTH_T_START="${GROWTH_T_START-}"
GROWTH_T_END="${GROWTH_T_END-}"

# CASE:run-folder. Add a run here once it has finished.
RUNS=(
    "firehose_bimaxwellian_strong_bigbox40:psc_firehose_bimaxwellian_strong_bigbox40_12062436"
    "firehose_bikappa3_bigbox40:psc_firehose_bikappa3_bigbox40_11657093"
    "mirror_bimaxwellian_moderate:psc_mirror_bimaxwellian_moderate_11596993"
    "mirror_bikappa3_moderate:psc_mirror_bikappa3_moderate_11618877"
    "mirror_bikappa5_moderate:psc_mirror_bikappa5_moderate_12063822"
)

# Relative cost of each stage (for longest-first scheduling) at 576^2;
# 1152^2 runs are weighted x4. Only the order depends on these numbers.
# (Plain functions and files instead of associative arrays: the script runs
# on any bash >= 3.2, whatever the node image ships.)
stage_cost() {
    case "$1" in
        physics) echo 10 ;; fields) echo 8 ;; spectrum) echo 7 ;;
        dispersion|diamagnetic|structures) echo 6 ;;
        growth-map-par|growth-map-perp|polarization|energy-exchange) echo 5 ;;
        brazil|heatflux) echo 4 ;; residuals|estimators|particles|vdf) echo 3 ;;
        *) echo 1 ;;
    esac
}
COMMON_STAGES=(physics fields spectrum dispersion growth-map-par growth-map-perp
               polarization diamagnetic structures energy-exchange brazil heatflux
               residuals estimators particles validate)
MIRROR_EXTRA_STAGES=(vdf)
# No separate `energy` stage: `physics` already writes the DiagEnergies
# budget into the same folder, and two steps must not write one file.

STATE_DIR="$LOG_DIR/runs"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$STATE_DIR"
say() { echo "[$(date '+%F %T')] $*"; }
run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: $*"; else "$@"; fi; }

say "reanalysis_v5 job $JOB_ID on ${SLURM_JOB_NUM_NODES:-?} nodes: ${SLURM_JOB_NODELIST:-local}"
say "results -> $NEW_ROOT ; logs -> $LOG_DIR ; clean=$CLEAN dry_run=$DRY_RUN"

# =====================================================================
# 0) The checkout must contain the v5 pipeline. Nothing is touched if not.
# =====================================================================
if [ ! -f growth_fit.py ] || ! grep -q '"analysis_conventions_version": 5' write_analysis_manifest.py; then
    echo "ERROR: $REPO is older than the v5 analysis pipeline. Run 'git pull' there first." >&2
    exit 2
fi
say "code: $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null || echo 'git unavailable')"

# =====================================================================
# 1) Preflight on a compute node: unit tests + synthetic end-to-end run.
#    Proves the COSMA Python environment runs the new code before any
#    old result is deleted.
# =====================================================================
say "preflight: unit tests and synthetic pipeline"
if ! run srun --nodes=1 --ntasks=1 --exclusive --job-name=preflight \
        env MPLCONFIGDIR="/tmp/psc-mpl-$JOB_ID" "$PY" -m unittest \
        test_growth_fit test_new_diagnostics test_physical_consistency \
        test_energy_conservation test_pipeline_synthetic \
        > "$LOG_DIR/preflight.log" 2>&1; then
    echo "ERROR: preflight failed (see $LOG_DIR/preflight.log); nothing was deleted." >&2
    exit 3
fi
say "preflight OK"

# =====================================================================
# 2) Resolve every run: folder, format (.bp / .h5), completeness.
# =====================================================================
CASES=()
last_field_step() {
    find "$1" -maxdepth 1 \( -name 'pfd.*.bp' -o -name 'pfd.*_p*.h5' \) 2>/dev/null \
        | sed -E 's|.*/pfd\.0*([0-9]+)[._].*|\1|' | sort -n | tail -1
}
for entry in "${RUNS[@]}"; do
    case_name="${entry%%:*}"
    folder="${entry#*:}"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $case_name "* ]]; then
        continue
    fi
    dir="$RUN_ROOT/$folder"
    if [ ! -d "$dir" ]; then
        say "SKIP $case_name: $dir does not exist"
        continue
    fi
    read -r nmax fields_every < <(PSC_PROFILE="$case_name" PSC_ANALYSIS_DATA_DIR= \
        "$PY" -c 'import psc_units as u; print(u.NMAX, u.FIELDS_EVERY)')
    last=$(last_field_step "$dir")
    if [ -z "$last" ]; then
        say "SKIP $case_name: no field snapshots in $dir"
        continue
    fi
    if [ "$last" -lt $((nmax - fields_every)) ] && [ "$ALLOW_INCOMPLETE" != 1 ]; then
        say "SKIP $case_name: last field step $last < nmax $nmax (unfinished; ALLOW_INCOMPLETE=1 forces it)"
        continue
    fi
    if compgen -G "$dir/pfd.*_p*.h5" > /dev/null; then fmt=h5; else fmt=bp; fi
    echo "$dir $fmt ${folder##*_}" > "$STATE_DIR/$case_name"
    CASES+=("$case_name")
    say "run $case_name: $dir ($fmt, last step $last / nmax $nmax)"
done
if [ "${#CASES[@]}" -eq 0 ]; then
    echo "ERROR: no finished run to analyse." >&2
    exit 4
fi

case_vars() {   # fills MV with the make variables of one case
    local c="$1" dir fmt tag
    read -r dir fmt tag < "$STATE_DIR/$c"
    MV=("DATA_DIR=$dir" "CASE=$c" "RESULTS_ROOT=$NEW_ROOT" "RUN_TAG=$tag"
        "SNAPSHOT_EVERY=$SNAPSHOT_EVERY" "GIF_EVERY=$GIF_EVERY" "FIELDS_GIF=$FIELDS_GIF"
        "GROWTH_T_START=$GROWTH_T_START" "GROWTH_T_END=$GROWTH_T_END"
        "MPLCONFIGDIR=/tmp/psc-mpl-$JOB_ID")
    if [ "$fmt" = bp ]; then
        MV+=("FIELD_PATTERN=$dir/pfd.*.bp"
             "MOMENT_PATTERN=$dir/pfd_moments.*.bp"
             "PARTICLE_PATTERN=$dir/prt_${c}.*.bp")
    fi
}

safe_rm() {     # delete an analysis-results directory, never anything else
    local target="$1"
    case "$target" in
        "$RESULTS_BASE"/?*) ;;
        *) echo "REFUSING to delete $target (outside $RESULTS_BASE)" >&2; return 1 ;;
    esac
    [[ "$target" == *..* ]] && { echo "REFUSING to delete $target" >&2; return 1; }
    [ -e "$target" ] || return 0
    say "delete $(du -sh "$target" 2>/dev/null | cut -f1) $target"
    run rm -rf -- "$target"
}

# =====================================================================
# 3) Manifest + initial-condition check per run (in parallel). Old results
#    of a run are deleted only after its new manifest succeeded.
# =====================================================================
MPIDS=()
for c in "${CASES[@]}"; do
    safe_rm "$NEW_ROOT/$c"          # leftovers of an earlier attempt
    case_vars "$c"
    ( run srun --nodes=1 --ntasks=1 --exclusive --job-name="manifest:$c" \
          make manifest "${MV[@]}" > "$LOG_DIR/${c}.manifest.log" 2>&1 ) &
    MPIDS+=($!)
done
READY=()
i=0
for c in "${CASES[@]}"; do
    pid="${MPIDS[$i]}"
    i=$((i + 1))
    if wait "$pid"; then
        say "manifest OK: $c"
        READY+=("$c")
        if [ "$CLEAN" = 1 ]; then
            for old in "${OLD_ROOTS[@]}"; do safe_rm "$old/$c"; done
        fi
    else
        say "manifest FAILED: $c -> not analysed, old results kept (see $LOG_DIR/${c}.manifest.log)"
    fi
done
if [ "$CLEAN" = 1 ]; then
    for old in "${OLD_ROOTS[@]}"; do safe_rm "$old/comparison_physical"; done
    for stale in "$NEW_ROOT"/comparison_* "$NEW_ROOT"/kappa_evolution_*; do
        [ -e "$stale" ] && safe_rm "$stale"
    done
fi
if [ "${#READY[@]}" -eq 0 ]; then
    echo "ERROR: every manifest failed; nothing analysed." >&2
    exit 5
fi

# =====================================================================
# 4) All stages of all runs, longest first, <= MAX_PARALLEL at a time.
# =====================================================================
stage_step() {  # run one stage of one case; returns its exit code
    local c="$1" stage="$2" log="$LOG_DIR/${1}.${2}.log"
    local step=(srun --nodes=1 --ntasks=1 --exclusive --job-name="$stage:$c")
    case_vars "$c"
    case "$stage" in
        growth-map-par)  run "${step[@]}" make growth-map GROWTH_COMPONENT=parallel "${MV[@]}" ;;
        growth-map-perp) run "${step[@]}" make growth-map GROWTH_COMPONENT=perp "${MV[@]}" ;;
        polarization)
            if [[ "$c" == firehose_* ]]; then
                # Parallel-propagation theory exists for firehose/EMIC (not
                # for the oblique mirror): overlay it on the PIC dispersion.
                run "${step[@]}" make theory "${MV[@]}" &&
                run "${step[@]}" make polarization \
                    THEORY_CSV="$NEW_ROOT/$c/04_spectra/linear_theory.csv" "${MV[@]}"
            else
                run "${step[@]}" make polarization "${MV[@]}"
            fi ;;
        vdf)
            run "${step[@]}" make vdf-spatial "${MV[@]}" &&
            run "${step[@]}" make theory-liouville "${MV[@]}" ;;
        *) run "${step[@]}" make "$stage" "${MV[@]}" ;;
    esac > "$log" 2>&1
}

TASKS=()
for c in "${READY[@]}"; do
    scale=1
    [[ "$c" == *bigbox40 ]] && scale=4
    stages=("${COMMON_STAGES[@]}")
    [[ "$c" == mirror_* ]] && stages+=("${MIRROR_EXTRA_STAGES[@]}")
    for s in "${stages[@]}"; do
        TASKS+=("$(( $(stage_cost "$s") * scale )) $c $s")
    done
done
SORTED=()
while IFS= read -r line; do SORTED+=("$line"); done \
    < <(printf '%s\n' "${TASKS[@]}" | sort -k1,1nr)
TASKS=("${SORTED[@]}")
say "${#TASKS[@]} stage steps for ${#READY[@]} runs, <= $MAX_PARALLEL at a time"

for task in "${TASKS[@]}"; do
    read -r _ c s <<< "$task"
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        # bash >= 4.3 returns as soon as one step ends; older bash polls.
        wait -n 2>/dev/null || sleep "$POLL"
    done
    say "start $c / $s"
    ( stage_step "$c" "$s"; echo $? > "$STATUS_DIR/${c}.${s}.rc" ) &
done
wait
say "all stage steps finished"

stage_ok() { [ "$(cat "$STATUS_DIR/${1}.${2}.rc" 2>/dev/null)" = 0 ]; }

# =====================================================================
# 5) Comparisons (only controlled pairs; only if their inputs succeeded).
# =====================================================================
pd() { echo "$NEW_ROOT/$1/09_physical_diagnostics"; }

compare_series() {  # compare_series NAME LABEL=CASE... (all must have succeeded)
    local name="$1" pair label c cases_arg="" kev=()
    shift
    for pair in "$@"; do
        label="${pair%%=*}"
        c="${pair#*=}"
        if ! { [[ " ${READY[*]} " == *" $c "* ]] && stage_ok "$c" physics; }; then
            say "comparison $name: $c was not analysed successfully in this job; skipped"
            return 0
        fi
        cases_arg="$cases_arg $label=$(pd "$c")"
        kev+=(--case "$label=$(pd "$c")")
    done
    say "comparison $name:${cases_arg//$NEW_ROOT\//}"
    run srun --nodes=1 --ntasks=1 --exclusive --job-name="compare:$name" \
        make compare-physics PYTHON="$PY" MPLCONFIGDIR="/tmp/psc-mpl-$JOB_ID" \
        COMPARE_OUT="$NEW_ROOT/comparison_$name" COMPARE_CASES="${cases_arg# }" \
        > "$LOG_DIR/comparison_$name.log" 2>&1
    echo $? > "$STATUS_DIR/comparison.${name}-compare.rc"
    run srun --nodes=1 --ntasks=1 --exclusive --job-name="kappa_evol:$name" \
        env MPLCONFIGDIR="/tmp/psc-mpl-$JOB_ID" "$PY" kappa_evolution.py "${kev[@]}" \
        --outdir "$NEW_ROOT/kappa_evolution_$name" \
        > "$LOG_DIR/kappa_evolution_$name.log" 2>&1
    echo $? > "$STATUS_DIR/comparison.${name}-kappa-evolution.rc"
}

compare_series mirror_moderate_kappa \
    "bi-Maxwellian=mirror_bimaxwellian_moderate" \
    "kappa5=mirror_bikappa5_moderate" "kappa3=mirror_bikappa3_moderate"
compare_series firehose_strong_40di_kappa \
    "bi-Maxwellian=firehose_bimaxwellian_strong_bigbox40" "kappa3=firehose_bikappa3_bigbox40"

# =====================================================================
# 6) Summary
# =====================================================================
SUMMARY="$NEW_ROOT/REANALYSIS_SUMMARY_${JOB_ID}.txt"
mkdir -p "$NEW_ROOT"
fail=0
{
    echo "reanalysis_v5 job $JOB_ID  finished $(date '+%F %T')"
    echo "code: $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null)"
    echo "results: $NEW_ROOT   logs: $LOG_DIR"
    echo
    printf '%-42s %-18s %s\n' CASE STAGE STATUS
    for rc_file in "$STATUS_DIR"/*.rc; do
        [ -e "$rc_file" ] || continue
        name="$(basename "$rc_file" .rc)"
        rc="$(cat "$rc_file")"
        [ "$rc" = 0 ] && status=OK || { status="FAILED (rc=$rc)"; fail=1; }
        printf '%-42s %-18s %s\n' "${name%.*}" "${name##*.}" "$status"
    done
    echo
    echo "Not analysed in this job: firehose_bimaxwellian_moderate_bigbox40, whistler_bimaxwellian_strong_mr800 (unfinished)."
    for c in "${CASES[@]}"; do
        [[ " ${READY[*]} " == *" $c "* ]] || echo "Manifest failed (old results kept): $c"
    done
} > "$SUMMARY"          # no pipe here: `fail` must survive the block
cat "$SUMMARY"
say "summary: $SUMMARY (exit $fail)"
exit "$fail"
