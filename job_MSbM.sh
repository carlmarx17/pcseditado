#!/bin/bash -l
#
# =====================================================================
#  Job SLURM: psc_M_S_bM  — COSMA7-rp  (nodos libres)
#  Submision: sbatch job_MSbM.sh
# =====================================================================

# --- Identificación del job ---
#SBATCH --job-name=psc_M_S_bM

# --- Salidas ---
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/output_MSbM.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/error_MSbM.%J.err

# --- Partición y cuenta ---
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433

# --- Recursos ---
# cosma7 nodes: 28 cores, ~256 GB RAM
# 32 nodos × 28 cores = 896 tareas  (encaja justo, sin saturar RAM)
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=28
#SBATCH --exclusive

# --- Tiempo máximo (cosma7-rp permite hasta 72h) ---
#SBATCH --time=72:00:00

# --- Directorio de trabajo (en Lustre, no en $HOME) ---
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/build/src

# --- Notificación por correo ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

# =====================================================================
# Entorno de módulos
# =====================================================================
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

# ADIOS2 instalado en $HOME
export ADIOS2_DIR=$HOME/adios2
export PATH=$ADIOS2_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ADIOS2_DIR/lib:$LD_LIBRARY_PATH

# =====================================================================
# Crear directorio de logs si no existe
# =====================================================================
mkdir -p /cosma7/data/dp433/dc-mart18/logs

# =====================================================================
# Diagnóstico útil al inicio del job
# =====================================================================
echo "============================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Job Name    : $SLURM_JOB_NAME"
echo " Nodos       : $SLURM_JOB_NODELIST"
echo " Total tasks : $SLURM_NTASKS"
echo " Inicio      : $(date)"
echo "============================================"

# =====================================================================
# Ejecución
# =====================================================================
mpirun -np $SLURM_NTASKS \
       --bind-to core \
       --map-by socket \
       ./psc_M_S_bM

echo "============================================"
echo " Fin: $(date)"
echo "============================================"
