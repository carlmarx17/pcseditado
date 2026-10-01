#!/bin/bash -l
#
# =====================================================================
#  reanalysis_v6_all.sh — run the v6 analysis pipeline (run_pipeline.py)
#  on every FINISHED anisotropy run, spread over up to 20 COSMA7 nodes.
#
#  Why: revision 6 of CodeforAnalisys (analysis conventions version 6)
#  changes the growth reference (modal fit instead of the biased domain rms),
#  VDF coordinates, structure definitions and uncertainty, and adds the
#  evidence report and the energy audit. The v5 products stay untouched as
#  the historical baseline: nothing is deleted by this job.
#  reanalysis_v5_all.sh refuses a v6 checkout on purpose.
#
#  Runs analysed (anisotropy_adios2/): the same five as v5
#    firehose_bimaxwellian_strong_bigbox40    psc_..._12062436
#    firehose_bikappa3_bigbox40               psc_..._11657093
#    mirror_bimaxwellian_moderate             psc_..._11596993
#    mirror_bikappa3_moderate                 psc_..._11618877
#    mirror_bikappa5_moderate                 psc_..._12063822
#  A run whose last field snapshot is short of the profile's nmax is skipped
#  (ALLOW_INCOMPLETE=1 forces it).
#
#  Order of work:
#    0. the checkout must contain the v6 pipeline (git pull first);
#    1. preflight on a compute node: the full test suite (pytest), which
#       includes the synthetic end-to-end run;
#    2. completeness check of every run;
#    3. one run_pipeline.py per run, all in parallel. Each runner owns its
#       results directory (lock, atomic stage records, logs) and launches
#       its stages as `srun` steps, one node per step, longest first:
#       manifest preflight -> physics -> spectral (with the accepted linear
#       phase), every other stage alongside. Mirror runs add the Liouville
#       closures and the parallel theory of the competing ion-cyclotron
#       branch; firehose runs the parallel linear theory for the
#       polarization overlay.
#    4. comparisons of the controlled series (only the distribution
#       changes), only between runs whose physics stage passed:
#         mirror moderate 20 d_i:  bi-Maxwellian / kappa 5 / kappa 3
#         firehose strong 40 d_i:  bi-Maxwellian / kappa 3
#       each with kappa_evolution.py and kappa_dynamics.py (kappa(t) against
#       the fluctuation energy and the local |B|, paired with the isotropic
#       controls present in the tree; a later job that analyses a control
#       re-runs the comparisons of its series);
#    5. evidence report over all runs (quality_report.py, with the energy
#       audit across runs) in <NEW_ROOT>/quality_report/index.html;
#    6. summary: <NEW_ROOT>/REANALYSIS_SUMMARY_<jobid>.txt
#
#  Submit (from the repository root on COSMA, after `git pull`):
#    sbatch cosma_jobs/analisis/reanalysis_v6_all.sh
#
#  Useful overrides (no editing needed):
#    sbatch --export=ALL,ONLY="mirror_bikappa5_moderate" <script>   # subset
#    sbatch --export=ALL,RESUME=1 <script>      # continue an interrupted job
#    sbatch --export=ALL,DRY_RUN=1 <script>     # print, run nothing
#    sbatch --export=ALL,GROWTH_T_START=5,GROWTH_T_END=20 <script>
#    sbatch --nodes=10 --ntasks=10 <script>     # fewer nodes, same result
#    sbatch --export=ALL,EXTRA_RUNS="case:folder ..." <script>   # add runs,
#        e.g. the isotropic controls psc_mirror_*_isotropic once finished
# =====================================================================

# --- Job identification ---
#SBATCH --job-name=reanalysis_v6

# --- Output ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/reanalysis_v6.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/reanalysis_v6.%J.err

# --- Partition and account ---
# cosma7-rp: same node pool as cosma7-rp-pauper, 72 h limit instead of 24 h.
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
# v6c: the final analysis tree (v6 and v6b stay as they are).
NEW_ROOT="${NEW_ROOT:-$REPO/analysis_results/v6c}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="/cosma7/data/dp433/dc-mart18/logs/reanalysis_v6_${JOB_ID}"
PY="${PYTHON:-$REPO/.venv/bin/python}"

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
ONLY="${ONLY:-}"
NODES="${SLURM_JOB_NUM_NODES:-20}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
GIF_EVERY="${GIF_EVERY:-10000}"
FIELDS_GIF="${FIELDS_GIF-1}"
GROWTH_T_START="${GROWTH_T_START-}"
GROWTH_T_END="${GROWTH_T_END-}"
STEP=(srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task="${SLURM_CPUS_PER_TASK:-28}")
export MPLCONFIGDIR="/tmp/psc-mpl-$JOB_ID" PSC_FIG_THEME=paper

# CASE:run-folder. Add a run here once it has finished, or pass it in
# EXTRA_RUNS (space-separated CASE:folder entries) without editing.
RUNS=(
    "firehose_bimaxwellian_strong_bigbox40:psc_firehose_bimaxwellian_strong_bigbox40_12062436"
    "firehose_bikappa3_bigbox40:psc_firehose_bikappa3_bigbox40_11657093"
    "mirror_bimaxwellian_moderate:psc_mirror_bimaxwellian_moderate_11596993"
    "mirror_bikappa3_moderate:psc_mirror_bikappa3_moderate_11618877"
    "mirror_bikappa5_moderate:psc_mirror_bikappa5_moderate_12063822"
)
for extra in ${EXTRA_RUNS:-}; do
    case "$extra" in
        ?*:?*) RUNS+=("$extra") ;;
        *) echo "ERROR: EXTRA_RUNS entry '$extra' is not CASE:folder" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR" "$NEW_ROOT"
say() { echo "[$(date '+%F %T')] $*"; }
run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: $*"; else "$@"; fi; }

say "reanalysis_v6 job $JOB_ID on $NODES nodes: ${SLURM_JOB_NODELIST:-local}"
say "results -> $NEW_ROOT ; logs -> $LOG_DIR ; resume=$RESUME dry_run=$DRY_RUN"

# =====================================================================
# 0) The checkout must contain the v6 pipeline. Nothing is touched if not.
# =====================================================================
if ! grep -q '^CONVENTIONS_VERSION = 6' analysis_contract.py 2>/dev/null \
        || ! grep -q -- '--launcher' run_pipeline.py 2>/dev/null; then
    echo "ERROR: $REPO is older than the v6 analysis pipeline. Run 'git pull' there first." >&2
    exit 2
fi
say "code: $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null || echo 'git unavailable')"

# =====================================================================
# 1) Preflight on a compute node: the whole test suite (it includes the
#    synthetic end-to-end run of every diagnostic script).
# =====================================================================
if ! "$PY" -c 'import pytest' 2>/dev/null; then
    echo "ERROR: pytest missing in $PY; run '$PY -m pip install -r requirements.txt'." >&2
    exit 3
fi
say "preflight: test suite"
if ! run "${STEP[@]}" --job-name=preflight "$PY" -m pytest -q -p no:cacheprovider . \
        > "$LOG_DIR/preflight.log" 2>&1; then
    echo "ERROR: preflight failed (see $LOG_DIR/preflight.log); nothing was analysed." >&2
    exit 3
fi
say "preflight OK: $(tail -1 "$LOG_DIR/preflight.log")"

# =====================================================================
# 2) Resolve every run: folder, format (.bp / .h5), completeness.
# =====================================================================
last_field_step() {
    # "$1/": follow the run folder when it is a symbolic link.
    find "$1/" -maxdepth 1 \( -name 'pfd.*.bp' -o -name 'pfd.*_p*.h5' \) 2>/dev/null \
        | sed -E 's|.*/pfd\.0*([0-9]+)[._].*|\1|' | sort -n | tail -1
}
CASES=(); DIRS=(); FORMATS=(); SKIPPED=()
skip() { say "SKIP $1"; SKIPPED+=("$1"); }
for entry in "${RUNS[@]}"; do
    case_name="${entry%%:*}"
    folder="${entry#*:}"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $case_name "* ]]; then
        continue
    fi
    dir="$RUN_ROOT/$folder"
    if [ ! -d "$dir" ]; then
        skip "$case_name: $dir does not exist"
        continue
    fi
    read -r nmax fields_every < <(PSC_PROFILE="$case_name" PSC_ANALYSIS_DATA_DIR= \
        "$PY" -c 'import psc_units as u; print(u.NMAX, u.FIELDS_EVERY)')
    last=$(last_field_step "$dir")
    if [ -z "$last" ]; then
        skip "$case_name: no field snapshots in $dir"
        continue
    fi
    if [ "$last" -lt $((nmax - fields_every)) ] && [ "$ALLOW_INCOMPLETE" != 1 ]; then
        skip "$case_name: last field step $last < nmax $nmax (unfinished; ALLOW_INCOMPLETE=1 forces it)"
        continue
    fi
    if [ "$RESUME" != 1 ] && [ -e "$NEW_ROOT/$case_name" ]; then
        skip "$case_name: $NEW_ROOT/$case_name exists (RESUME=1 continues it; nothing is overwritten)"
        continue
    fi
    if compgen -G "$dir/pfd.*_p*.h5" > /dev/null; then fmt=h5; else fmt=bp; fi
    CASES+=("$case_name"); DIRS+=("$dir"); FORMATS+=("$fmt")
    say "run $case_name: $dir ($fmt, last step $last / nmax $nmax)"
done
if [ "${#CASES[@]}" -eq 0 ]; then
    echo "ERROR: no finished run to analyse." >&2
    exit 4
fi

# =====================================================================
# 3) One runner per run, in parallel; each spreads its stages over nodes.
# =====================================================================
JOBS_PER_RUN=$(( NODES / ${#CASES[@]} ))
[ "$JOBS_PER_RUN" -lt 1 ] && JOBS_PER_RUN=1
say "${#CASES[@]} runs, up to $JOBS_PER_RUN concurrent stages each"

PIDS=()
for i in "${!CASES[@]}"; do
    c="${CASES[$i]}"; dir="${DIRS[$i]}"; fmt="${FORMATS[$i]}"
    opts=(--make-option "RUN_TAG=${dir##*_}" --make-option "SNAPSHOT_EVERY=$SNAPSHOT_EVERY"
          --make-option "GIF_EVERY=$GIF_EVERY" --make-option "FIELDS_GIF=$FIELDS_GIF")
    [ -n "$GROWTH_T_START" ] && opts+=(--make-option "GROWTH_T_START=$GROWTH_T_START")
    [ -n "$GROWTH_T_END" ] && opts+=(--make-option "GROWTH_T_END=$GROWTH_T_END")
    if [ "$fmt" = bp ]; then
        opts+=(--make-option "FIELD_PATTERN=$dir/pfd.*.bp"
               --make-option "MOMENT_PATTERN=$dir/pfd_moments.*.bp"
               --make-option "PARTICLE_PATTERN=$dir/prt_${c}.*.bp")
    fi
    stages=(manifest residuals physics brazil spectral structures energy-exchange estimators
            heatflux particles validate diamagnetic fields vdf-spatial)
    case "$c" in
        # Isotropic controls: no drive, no branch to solve; a failed theory
        # stage would also hold back their spectral stage.
        *_isotropic) ;;
        # mirror: the theory stage solves the competing parallel ion-cyclotron
        # branch (the mode that grows in these runs), never the mirror mode.
        mirror_*)   stages+=(theory-liouville theory) ;;
        firehose_*) stages+=(theory) ;;
    esac
    resume=(); [ "$RESUME" = 1 ] && resume=(--resume)
    say "start runner $c"
    ( run "$PY" run_pipeline.py --data-dir "$dir" --case "$c" --results-root "$NEW_ROOT" \
          --jobs "$JOBS_PER_RUN" --launcher "${STEP[*]}" --keep-going ${resume[@]+"${resume[@]}"} \
          --stages "${stages[@]}" "${opts[@]}" > "$LOG_DIR/${c}.runner.log" 2>&1
      echo $? > "$LOG_DIR/${c}.runner.rc" ) &
    PIDS+=($!)
done
wait "${PIDS[@]}"
say "all runners finished"

# Resonant anisotropy A(v_par) of the ion-driven mirror runs and their
# isotropic controls (particle + field snapshots; outside the runner because
# it is not a stage of the validation matrix).
PIDS=()
for i in "${!CASES[@]}"; do
    c="${CASES[$i]}"; dir="${DIRS[$i]}"
    [[ "$c" == mirror_* ]] || continue
    ( run "${STEP[@]}" --job-name="res_aniso:$c" env PSC_PROFILE="$c" PSC_ANALYSIS_DATA_DIR="$dir" \
          "$PY" resonant_anisotropy.py measure --data-dir "$dir" --outdir "$NEW_ROOT/$c/03_particles" \
          > "$LOG_DIR/${c}.resonant_anisotropy.log" 2>&1
      echo $? > "$LOG_DIR/${c}.resonant-anisotropy.rc" ) &
    PIDS+=($!)
done
[ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
say "resonant anisotropy finished"

stage_passed() {  # stage_passed CASE STAGE
    "$PY" - "$NEW_ROOT/$1/pipeline.json" "$2" <<'EOF' 2>/dev/null
import json, sys
state = json.load(open(sys.argv[1]))
sys.exit(0 if state['stages'].get(sys.argv[2], {}).get('execution_status') == 'PASS' else 1)
EOF
}

# =====================================================================
# 4) Comparisons (only controlled series; only runs whose physics passed).
# =====================================================================
pd() { echo "$NEW_ROOT/$1/09_physical_diagnostics"; }
compare_series() {  # compare_series NAME LABEL=CASE... (all must have succeeded)
    local name="$1" pair label c cases_arg="" kev=() roots=() fresh=0
    shift
    for pair in "$@"; do    # only when this job analysed a member or its isotropic control
        c="${pair#*=}"
        [[ " ${CASES[*]} " == *" $c "* ]] && fresh=1
        [[ " ${CASES[*]} " == *" ${c%_*}_isotropic "* ]] && fresh=1
    done
    if [ "$fresh" = 0 ]; then
        say "comparison $name: no member analysed in this job; kept as is"
        return 0
    fi
    for pair in "$@"; do
        label="${pair%%=*}"
        c="${pair#*=}"
        if ! stage_passed "$c" physics; then
            say "comparison $name: $c has no passed physics stage in $NEW_ROOT; skipped"
            return 0
        fi
        cases_arg="$cases_arg $label=$(pd "$c")"
        kev+=(--case "$label=$(pd "$c")")
        roots+=("$NEW_ROOT/$c")
    done
    say "comparison $name:${cases_arg//$NEW_ROOT\//}"
    run "${STEP[@]}" --job-name="compare:$name" \
        make compare-physics PYTHON="$PY" \
        COMPARE_OUT="$NEW_ROOT/comparison_$name" COMPARE_CASES="${cases_arg# }" \
        > "$LOG_DIR/comparison_$name.log" 2>&1
    echo $? > "$LOG_DIR/comparison_${name}.compare.rc"
    run "${STEP[@]}" --job-name="kappa_evol:$name" \
        "$PY" kappa_evolution.py "${kev[@]}" --outdir "$NEW_ROOT/kappa_evolution_$name" \
        > "$LOG_DIR/kappa_evolution_$name.log" 2>&1
    echo $? > "$LOG_DIR/comparison_${name}.kappa-evolution.rc"
    # kappa(t) against the fluctuation energy and the local |B|; isotropic
    # controls analysed in $NEW_ROOT (*_isotropic) are paired automatically.
    run "${STEP[@]}" --job-name="kappa_dyn:$name" \
        "$PY" kappa_dynamics.py "${roots[@]}" --outdir "$NEW_ROOT/kappa_dynamics_$name" \
        > "$LOG_DIR/kappa_dynamics_$name.log" 2>&1
    echo $? > "$LOG_DIR/comparison_${name}.kappa-dynamics.rc"
    # Publication figures of the series (ion-cyclotron mode, gamma vs kappa
    # against theory, trajectories, resonance, local kappa): mirror series.
    if [[ "$name" == mirror_* ]]; then
        run "${STEP[@]}" --job-name="paper_fig:$name" \
            "$PY" paper_figures.py "${roots[@]}" --outdir "$NEW_ROOT/paper_figures/$name" \
            > "$LOG_DIR/paper_figures_$name.log" 2>&1
        echo $? > "$LOG_DIR/comparison_${name}.paper-figures.rc"
        # Linear theory re-evaluated with the moments and kappa measured at
        # each time, against the growth the mode has then (products only).
        run "${STEP[@]}" --job-name="traj_lin:$name" \
            "$PY" trajectory_linear_theory.py "${roots[@]}" --outdir "$NEW_ROOT/trajectory_linear_theory_$name" \
            > "$LOG_DIR/trajectory_linear_theory_$name.log" 2>&1
        echo $? > "$LOG_DIR/comparison_${name}.trajectory-linear-theory.rc"
        # Where in parallel velocity the anisotropy is left (needs the
        # resonant_anisotropy.csv of each run, written above).
        run "${STEP[@]}" --job-name="res_aniso_fig:$name" \
            "$PY" resonant_anisotropy.py plot "${roots[@]}" --outdir "$NEW_ROOT/resonant_anisotropy_$name" \
            > "$LOG_DIR/resonant_anisotropy_$name.log" 2>&1
        echo $? > "$LOG_DIR/comparison_${name}.resonant-anisotropy.rc"
    fi
}
compare_series mirror_moderate_kappa \
    "bi-Maxwellian=mirror_bimaxwellian_moderate" \
    "kappa5=mirror_bikappa5_moderate" "kappa3=mirror_bikappa3_moderate"
compare_series firehose_strong_40di_kappa \
    "bi-Maxwellian=firehose_bimaxwellian_strong_bigbox40" "kappa3=firehose_bikappa3_bigbox40"

# =====================================================================
# 5) Evidence report and energy audit over every run of the v6 tree (not
#    only this job's), so that each run meets its isotropic control.
# =====================================================================
REPORT_RUNS=()
for record in "$NEW_ROOT"/*/pipeline.json; do
    [ -f "$record" ] && REPORT_RUNS+=("$(dirname "$record")")
done
if [ "${#REPORT_RUNS[@]}" -gt 0 ]; then
    run "$PY" quality_report.py "${REPORT_RUNS[@]}" --outdir "$NEW_ROOT/quality_report" \
        > "$LOG_DIR/quality_report.log" 2>&1
    echo $? > "$LOG_DIR/quality_report.rc"
fi

# =====================================================================
# 6) Summary
# =====================================================================
SUMMARY="$NEW_ROOT/REANALYSIS_SUMMARY_${JOB_ID}.txt"
{
    echo "reanalysis_v6 job $JOB_ID  finished $(date '+%F %T')"
    echo "code: $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null)"
    echo "results: $NEW_ROOT   logs: $LOG_DIR"
    echo "Execution status per stage (scientific status: quality_report/index.html)"
    echo
    if [ "${#SKIPPED[@]}" -gt 0 ]; then
        echo "Runs not analysed by this job:"
        printf '   %s\n' "${SKIPPED[@]}"
        echo
    fi
    for c in "${CASES[@]}"; do
        echo "== $c (runner rc=$(cat "$LOG_DIR/${c}.runner.rc" 2>/dev/null || echo '?'))"
        "$PY" - "$NEW_ROOT/$c/pipeline.json" <<'EOF' 2>/dev/null || echo "   no pipeline record"
import json, sys
for stage, e in json.load(open(sys.argv[1]))['stages'].items():
    print(f"   {stage:18s} {e.get('execution_status', '?'):8s} {e.get('elapsed_seconds', 0) / 3600:6.2f} h  {e.get('reason', '')}")
EOF
    done
    echo
    for rc_file in "$LOG_DIR"/comparison_*.rc "$LOG_DIR"/quality_report.rc; do
        [ -e "$rc_file" ] || continue
        rc="$(cat "$rc_file")"
        printf '%-60s %s\n' "$(basename "$rc_file" .rc)" "$([ "$rc" = 0 ] && echo OK || echo "FAILED (rc=$rc)")"
    done
    if [ -f "$NEW_ROOT/quality_report/validation_matrix.json" ]; then
        echo
        "$PY" - "$NEW_ROOT/quality_report/validation_matrix.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
for r in m['runs']:
    print(f"{r['run']:42s} scientific status {r['scientific_status']}")
    for c in r['checks']:
        if c['check'] == 'energy_baseline':
            print(f"   baseline-corrected energy: {c['status']} -- {c['reason']}")
for g in m.get('energy_groups', []):
    print(g['interpretation'])
EOF
    fi
} > "$SUMMARY"
cat "$SUMMARY"
