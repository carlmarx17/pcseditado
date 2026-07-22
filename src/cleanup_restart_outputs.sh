#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  cleanup_restart_outputs.sh --step STEP [--run-dir DIR] [--delete]

Lists, or deletes with --delete, PSC output files whose timestep is greater
than STEP. Use this before restarting in the same run directory from
checkpoint_STEP.bp, so new output does not overwrite stale later files.

Examples:
  ./cleanup_restart_outputs.sh --step 300000 --run-dir /path/to/run
  ./cleanup_restart_outputs.sh --step 300000 --run-dir /path/to/run --delete

Matched output:
  checkpoint_<step>.bp
  pfd.<step>*.h5, pfd.<step>.xdmf
  pfd_moments.<step>*.h5, pfd_moments.<step>.xdmf
  tfd.<step>*.h5, tfd.<step>.xdmf
  tfd_moments.<step>*.h5, tfd_moments.<step>.xdmf
  prt_*.<step>.h5

Also removes temporal index files pfd.xdmf, pfd_moments.xdmf, tfd.xdmf, and
tfd_moments.xdmf because they may still reference deleted timesteps.
USAGE
}

step=""
run_dir="."
delete=0

while (($#)); do
  case "$1" in
    --step)
      step="${2:-}"
      shift 2
      ;;
    --run-dir)
      run_dir="${2:-}"
      shift 2
      ;;
    --delete)
      delete=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$step" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --step must be an integer timestep" >&2
  exit 2
fi

if [[ ! -d "$run_dir" ]]; then
  echo "ERROR: --run-dir does not exist: $run_dir" >&2
  exit 2
fi

step=$((10#$step))
mapfile -d '' matches < <(
  find "$run_dir" -maxdepth 1 \( -type f -o -type d \) -print0 |
    while IFS= read -r -d '' path; do
      name=${path##*/}
      ts=""

      case "$name" in
        checkpoint_*.bp)
          ts=${name#checkpoint_}
          ts=${ts%.bp}
          ;;
        pfd.*|pfd_moments.*|tfd.*|tfd_moments.*)
          ts=${name#*.}
          ts=${ts%%_*}
          ts=${ts%%.xdmf}
          ;;
        prt_*.h5)
          ts=${name%.h5}
          ts=${ts##*.}
          ;;
      esac

      if [[ "$ts" =~ ^[0-9]+$ ]] && ((10#$ts > step)); then
        printf '%s\0' "$path"
      fi
    done
)

for index in pfd.xdmf pfd_moments.xdmf tfd.xdmf tfd_moments.xdmf; do
  if [[ -e "$run_dir/$index" ]]; then
    matches+=("$run_dir/$index")
  fi
done

if ((${#matches[@]} == 0)); then
  echo "No files matched for timestep > $step in $run_dir"
  exit 0
fi

printf 'Matched %d path(s) for cleanup in %s:\n' "${#matches[@]}" "$run_dir"
printf '  %s\n' "${matches[@]}"

if ((delete)); then
  rm -rf -- "${matches[@]}"
  echo "Deleted ${#matches[@]} path(s)."
else
  echo
  echo "Dry run only. Re-run with --delete to remove these paths."
fi
