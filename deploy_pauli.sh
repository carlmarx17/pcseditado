#!/bin/bash
# ============================================================
#  deploy_pauli.sh
#  Sube el ejecutable y el script SLURM a pauli y lanza el job
#  Uso: ./deploy_pauli.sh
# ============================================================

set -euo pipefail

# --- Configuración ---
REMOTE_USER="cmartinezsi"
REMOTE_HOST="perseus"                                      # login node
REMOTE_DIR="/homes/observatorio/cmartinezsi/pcs_run"
EXECUTABLE="./psc_mirror_maxwellian_2k"
SLURM_SCRIPT="psc_mirror_max2k.slurm"

# --- Colores ---
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Verificaciones locales ---
info "Verificando archivos locales..."
[ -f "$EXECUTABLE" ]    || error "No se encuentra $EXECUTABLE"
[ -f "$SLURM_SCRIPT" ]  || error "No se encuentra $SLURM_SCRIPT"
info "Ejecutable : $EXECUTABLE  ($(du -h $EXECUTABLE | cut -f1))"
info "SLURM script: $SLURM_SCRIPT"

# --- Crear directorio remoto si no existe ---
info "Asegurando directorio remoto $REMOTE_DIR en $REMOTE_HOST..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# --- Copiar archivos ---
info "Copiando archivos → ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
scp -v "$EXECUTABLE" "$SLURM_SCRIPT" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# --- Permisos en remoto ---
info "Ajustando permisos del ejecutable..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "chmod +x ${REMOTE_DIR}/psc_mirror_maxwellian_2k"

# --- Enviar job ---
info "Enviando job SLURM desde pauli..."
JOB_OUTPUT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ${REMOTE_DIR} && sbatch ${SLURM_SCRIPT}")

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  $JOB_OUTPUT${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""

# Extraer JOBID
JOBID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')

if [ -n "$JOBID" ]; then
    info "Monitoreo rápido del job $JOBID:"
    ssh "${REMOTE_USER}@${REMOTE_HOST}" \
        "sleep 3 && squeue --me --job=${JOBID} 2>/dev/null || squeue --me"
    echo ""
    info "Para seguir el output:"
    echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "  tail -f ${REMOTE_DIR}/psc_mirror_max2k_${JOBID}.out"
fi
