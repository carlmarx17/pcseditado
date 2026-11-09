#!/bin/bash -l
#SBATCH --job-name=tesis_mirror
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --output=tesis_mirror_%j.log

set -eo pipefail
site=/cosma7/data/dp433/dc-mart18
code=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
data=${RUN_ROOT:-$site/anisotropy_adios2}
out=${OUT:-$site/pcseditado/analysis_results/tesis_mirror_${SLURM_JOB_ID:-local}}
py=${PYTHON:-$site/pcseditado/.venv/bin/python}
cd "$code/CodeforAnalisys"
source "$code/src/cosma_adios2_env.sh"
set -u
export MPLBACKEND=Agg PSC_FIG_THEME=paper OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PSC_ANALYSIS_CONFIG PSC_ANALYSIS_DATA_DIR PSC_ANALYSIS_RUN_ID PSC_PROFILE
mkdir -p "$out/logs"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/psc-mpl-${SLURM_JOB_ID:-local}"
step=(srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK:-28}")
names=(mirror_bimaxwellian_moderate mirror_bikappa5_moderate mirror_bikappa3_moderate)
ids=(11596993 12063822 11618877)
roots=() controls=() dirs=() cases=() destinations=()
shopt -s nullglob
for i in "${!names[@]}"; do
    case_name=${names[i]}
    folder="$data/psc_${case_name}_${ids[i]}"
    [[ -d "$folder" ]] || { echo "Missing production directory $folder"; exit 2; }
    roots+=("$out/$case_name")
    dirs+=("$folder"); cases+=("$case_name"); destinations+=("$out")
    control=${case_name%_moderate}_isotropic
    key="CONTROL_${i}"
    if [[ -n ${!key:-} ]]; then
        found=("${!key}")
        [[ -d ${found[0]} ]] || { echo "Missing control ${found[0]}"; exit 2; }
    else
        found=()
        for candidate in "$data"/psc_"${control}"_*; do
            snapshots=("$candidate"/pfd.*.bp "$candidate"/pfd.*.h5)
            if (( ${#snapshots[@]} )); then found+=("$candidate"); fi
        done
    fi
    if (( ${#found[@]} > 1 )); then
        echo "Multiple histories for $control. Set $key to the chosen absolute directory."; exit 2
    elif (( ${#found[@]} == 1 )); then
        dirs+=("${found[0]}"); cases+=("$control"); destinations+=("$out/controls")
        controls+=("$out/controls/$control")
    else
        echo "$control: no snapshots located; raw analysis only"
    fi
done

analyse() {
    local case_name=$1 folder=$2 dest=$3 bounds last expected
    local -a stages options bp h5
    bounds=$("$py" check_run_completion.py --case "$case_name" --data-dir "$folder")
    read -r last expected _ <<< "$bounds"
    [[ $last =~ ^[0-9]+$ && $expected =~ ^[0-9]+$ ]] || { echo "Invalid completion record: $bounds"; return 2; }
    echo "$case_name  $folder  last=$last expected=$expected"
    options=(--make-option "RUN_TAG=${folder##*_}" --make-option FIELDS_GIF=0)
    bp=("$folder"/pfd.*.bp); h5=("$folder"/pfd.*.h5)
    if (( ${#bp[@]} && ${#h5[@]} )); then echo "Mixed field formats in $folder"; return 2; fi
    if (( ${#bp[@]} )); then
        options+=(--make-option "FIELD_PATTERN=$folder/pfd.*.bp"
                  --make-option "MOMENT_PATTERN=$folder/pfd_moments.*.bp"
                  --make-option "PARTICLE_PATTERN=$folder/prt_${case_name}.*.bp")
    fi
    if [[ $case_name == *_isotropic ]]; then
        stages=(manifest residuals physics vdf-spatial)
    else
        (( last >= expected )) || { echo "Incomplete production history"; return 2; }
        stages=(manifest residuals physics brazil spectral structures energy-exchange estimators
                heatflux particles validate diamagnetic fields vdf-spatial theory theory-liouville)
        options+=(--make-option STRUCTURES_FLAGS=--attribution)
    fi
    "$py" run_pipeline.py --data-dir "$folder" --case "$case_name" --results-root "$dest" \
        --resume --jobs 1 --stages "${stages[@]}" "${options[@]}" || return
    if [[ $case_name == *_moderate ]]; then
        PSC_PROFILE="$case_name" PSC_ANALYSIS_DATA_DIR="$folder" "$py" resonant_anisotropy.py measure \
            --data-dir "$folder" --outdir "$dest/$case_name/03_particles" || return
    fi
}
export -f analyse
export py
pids=()
for i in "${!cases[@]}"; do
    "${step[@]}" bash -ec 'shopt -s nullglob; analyse "$@"' _ "${cases[i]}" "${dirs[i]}" "${destinations[i]}" \
        > "$out/logs/${cases[i]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then echo "Failed: ${cases[i]} (see $out/logs)"; failed=1; fi
done
(( failed == 0 )) || exit 1
review_args=(); dynamics_args=()
for control in "${controls[@]}"; do
    review_args+=(--isotropic-control "$control")
    dynamics_args+=(--control "$control")
done
"$py" review_mirror_energy.py "$out" --outdir "$out/review" "${review_args[@]}"
"$py" quality_report.py "${roots[@]}" --outdir "$out/quality_report"
"$py" resonant_anisotropy.py plot "${roots[@]}" --outdir "$out/resonant_anisotropy_mirror_moderate_kappa"
"$py" trajectory_linear_theory.py "${roots[@]}" --outdir "$out/trajectory_linear_theory_mirror_moderate_kappa"
"$py" kappa_dynamics.py "${roots[@]}" "${dynamics_args[@]}" --outdir "$out/kappa_dynamics_mirror_moderate_kappa"
"$py" window_vs_box_fluctuation.py "$out" --family mirror
"$py" kappa_frame_comparison.py "$out" --family mirror
"$py" paper_figures.py "${roots[@]}" --outdir "$out/paper_figures/mirror_moderate_kappa"
"$py" kappa_linear_survey.py --runs "${roots[@]}" --jobs "${SLURM_CPUS_PER_TASK:-1}" --outdir "$out/kappa_linear_survey"
"$py" wave_kappa_test.py "$out" --family mirror
echo "Review: $out/review/REVISION.md"
