#!/bin/bash

# Shared COSMA environment for PSC + ADIOS2 jobs. Source this file; do not run
# it directly. It intentionally avoids Conda and login-shell startup files.

unset BASH_ENV ENV

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

unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL

if [ -n "${PATH:-}" ]; then
  clean_path=
  old_ifs=$IFS
  IFS=:
  for path_entry in $PATH; do
    case "$path_entry" in
      "$HOME"/miniconda*|"$HOME"/anaconda*|*/miniconda*/bin|*/anaconda*/bin|*/conda*/bin|*/condabin)
        continue
        ;;
    esac
    if [ -z "$clean_path" ]; then
      clean_path="$path_entry"
    else
      clean_path="$clean_path:$path_entry"
    fi
  done
  IFS=$old_ifs
  PATH=$clean_path
  export PATH
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
  elif [ -x "$HOME/build_adios2_nohdf5/build/bin/adios2-config" ]; then
    ADIOS2_DIR="$HOME/build_adios2_nohdf5/build"
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

psc_mpi_run() {
  nproc=$1
  shift

  case "${PSC_LAUNCHER:-srun}" in
    srun)
      if [ -n "${PSC_SRUN_MPI_TYPE:-}" ]; then
        srun_mpi_type=$PSC_SRUN_MPI_TYPE
        if [ "$srun_mpi_type" = "pmi2" ] && [ "${PSC_ALLOW_SRUN_PMI2:-0}" != "1" ]; then
          echo "ERROR: PSC_SRUN_MPI_TYPE=pmi2 is unsafe with OpenMPI 5." >&2
          echo "       It can start singleton MPI ranks and trigger OOM." >&2
          echo "       Use a PMIx type, PSC_LAUNCHER=mpirun, or set PSC_ALLOW_SRUN_PMI2=1 knowingly." >&2
          return 2
        fi
      else
        srun_mpi_type=
        srun_mpi_list="$(srun --mpi=list 2>&1 || true)"
        normalized_mpi_list=" $(printf '%s' "$srun_mpi_list" | tr ',\n\t' '    ') "
        for candidate in pmix pmix_v4 pmix_v3; do
          case "$normalized_mpi_list" in
            *" $candidate "*)
              srun_mpi_type=$candidate
              break
              ;;
          esac
        done
        if [ -z "$srun_mpi_type" ]; then
          echo "WARNING: no PMIx-capable srun MPI type found; falling back to mpirun." >&2
          echo "         srun --mpi=pmi2 is not used because OpenMPI 5 may start singleton ranks." >&2
          echo "launcher=mpirun -np $nproc $*"
          mpirun -np "$nproc" "$@"
          return
        fi
      fi
      echo "launcher=srun --mpi=$srun_mpi_type -n $nproc $*"
      srun --mpi="$srun_mpi_type" -n "$nproc" "$@"
      ;;
    mpirun)
      echo "launcher=mpirun -np $nproc $*"
      mpirun -np "$nproc" "$@"
      ;;
    *)
      echo "ERROR: unsupported PSC_LAUNCHER=${PSC_LAUNCHER}" >&2
      echo "       Use PSC_LAUNCHER=srun or PSC_LAUNCHER=mpirun." >&2
      return 2
      ;;
  esac
}

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
