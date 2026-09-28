#!/bin/bash
# Preserve the previous segment before DiagEnergies opens diag.asc with "w".
set -euo pipefail

energy_run_dir=${1:-.}
if [[ -s "$energy_run_dir/diag.asc" ]]; then
  # BSD mktemp (macOS) only expands trailing X's, so the .asc suffix is added
  # with a rename; -n keeps an existing archive from being overwritten.
  energy_tmp=$(mktemp "$energy_run_dir/diag.archive.XXXXXXXX")
  energy_archive="$energy_tmp.asc"
  mv -n -- "$energy_tmp" "$energy_archive"
  cp -- "$energy_run_dir/diag.asc" "$energy_archive"
  echo "Preserved energy segment: $energy_archive"
fi
