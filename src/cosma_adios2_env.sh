#!/bin/bash

# Shared COSMA environment for PSC + ADIOS2 jobs. Source this file; do not run
# it directly. It intentionally avoids Conda and login-shell startup files.

if ! type module >/dev/null 2>&1; then
  for modules_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [ -r "$modules_init" ]; then
      # shellcheck source=/dev/null
      source "$modules_init"
      break
    fi
  done
fi

type module >/dev/null 2>&1 || {
  echo "ERROR: module command is not available" >&2
  return 1 2>/dev/null || exit 1
}

if [ -z "${HOME:-}" ]; then
  HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
  export HOME
fi

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

if [ -z "${ADIOS2_DIR:-}" ]; then
  if [ -x "$HOME/adios2/bin/adios2-config" ]; then
    ADIOS2_DIR="$HOME/adios2"
  elif [ -x "$HOME/adios2-nohdf5/bin/adios2-config" ]; then
    ADIOS2_DIR="$HOME/adios2-nohdf5"
  elif command -v adios2-config >/dev/null 2>&1; then
    ADIOS2_DIR="$(cd "$(dirname "$(command -v adios2-config)")/.." && pwd)"
  else
    ADIOS2_DIR="$HOME/adios2"
  fi
fi
export ADIOS2_DIR

if [ -z "${HDF5_ROOT:-}" ]; then
  HDF5_ROOT="/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4"
fi
export HDF5_ROOT

export PATH="$ADIOS2_DIR/bin:$PATH"
if [ -d "$HDF5_ROOT/lib" ]; then
  export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$ADIOS2_DIR/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$ADIOS2_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

command -v h5pcc >/dev/null 2>&1 || {
  echo "ERROR: h5pcc not found after loading parallel_hdf5/1.14.4" >&2
  return 1 2>/dev/null || exit 1
}

if [ "${COSMA_REQUIRE_ADIOS2:-1}" != "0" ]; then
  command -v adios2-config >/dev/null 2>&1 || {
    echo "ERROR: adios2-config not found. Set ADIOS2_DIR or install ADIOS2 in \$HOME/adios2." >&2
    return 1 2>/dev/null || exit 1
  }
fi
