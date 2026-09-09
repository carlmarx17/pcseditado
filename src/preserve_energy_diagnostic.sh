#!/bin/bash
# Preserve the previous segment before DiagEnergies opens diag.asc with "w".
set -euo pipefail

energy_run_dir=${1:-.}
if [[ -s "$energy_run_dir/diag.asc" ]]; then
  energy_archive=$(mktemp "$energy_run_dir/diag.archive.XXXXXXXX.asc")
  cp -- "$energy_run_dir/diag.asc" "$energy_archive"
  echo "Preserved energy segment: $energy_archive"
fi
