#!/bin/bash
# =============================================================================
# interactive_boltzmann.sh  —  Sesión interactiva en boltzmann para pruebas
#
# Cluster sinfo (as of 2026-04-10):
#   boltzmann,hercules7,...  32+ cores, ~62 GB RAM  ->  use 32 MPI tasks
#
# NOTA: boltzmann tiene 32+ cores. Se usan 32 ranks con np{1,4,8}=32.
#       El grid de prueba (mini: 512×512) cabe en qualquier config MPI divisible.
#
# Este script pide una sesión interactiva en boltzmann para:
#   - Verificar que el ejecutable compila y corre
#   - Probar la configuración ADIOS2/BP5 con pocos pasos
#   - Revisar la resolución lambda_De en stdout
#   - Medir uso de RAM y disco antes del run largo en feynman
#
# Uso:
#   bash scripts/interactive_boltzmann.sh
# =============================================================================

SCRATCH="/scratchsan/observatorio/cmartinezsi/pcseditado"

echo "Solicitando sesión interactiva en boltzmann (32 cores, 5h)..."
echo "Una vez dentro, ejecuta los comandos del bloque DENTRO_DE_BOLTZMANN."
echo ""

salloc \
  --job-name=interactive_boltzmann \
  --clusters=fisica \
  --partition=cpu.cecc \
  --nodelist=boltzmann \
  --qos=low \
  --nodes=1 \
  --ntasks=32 \
  --time=05:00:00

# ─────────────────────────────────────────────────────────────────────────────
# DENTRO_DE_BOLTZMANN: pega estos comandos una vez dentro de la sesión
# ─────────────────────────────────────────────────────────────────────────────
#
# module purge
# module load MPI/openmpi/4.1.1
# module load lang/gcc/9.2
#
# SCRATCH="/scratchsan/observatorio/cmartinezsi/pcseditado"
# mkdir -p "${SCRATCH}/Out1_test" && cd "${SCRATCH}/Out1_test"
#
# # Test rápido: solo 200 pasos con configuración reducida (32 ranks)
# # np{1,4,8}=32 funciona con el grid de producción 4096×4608
# # ya que 4096 es divisible por 4 y 4608 es divisible por 8.
# export ADIOS2_XML_CONFIG="${SCRATCH}/adios2cfg.xml"
# mpirun -np 32 "${SCRATCH}/build/src/psc_temp_aniso" \
#     --nmax 200 \
#     2>&1 | tee test_200steps.log
#
# # ADVERTENCIA: boltzmann y feynman-00 tienen distinto numero de cores.
# # El grid np{1,8,9}=72 es para feynman-00 (72 cores).
# # Con 32 ranks en boltzmann PSC usara np compatible (ajusta en setupGrid
# # si quieres correr en boltzmann de forma regular).
#
# # Verifica en el log:
# #   "pts/lambda_De = 3.07  (need >= 3)"   <- resolución OK
# #   "dt = 0.01545 => nmax=800000 covers..."
# #
# # Ver uso de RAM (en otra terminal ssh):
# #   watch -n 5 'free -h; df -h /scratchsan'
# #
# # Tamaño del output tras el test (dos snapshots):
# #   du -sh Out1_test/
# #   # Esperado: ~2 x 90 MB ZFP = ~180 MB para pfd.bp
# ─────────────────────────────────────────────────────────────────────────────
