# Revisión de ejecución de todos los análisis PSC

Revisión local del 14 de octubre de 2026. Incluye el runner, los diagnósticos,
la lectura de datos y el trabajo Slurm `cosma_jobs/analisis/reanalysis_v6_all.sh`.
No se ejecutaron ni modificaron las simulaciones o los datos reales de COSMA.

## Correcciones

- Los fallos al leer snapshots o procesarlos en paralelo ya no producen una
  serie parcial marcada como correcta. La etapa falla y conserva su log.
- El perfil solicitado prevalece sobre variables heredadas. La normalización
  magnética utiliza el B0 del perfil, incluidos los perfiles antiguos con B0=0.05.
- Un proceso que termina con código cero pero sin productos ya no recibe PASS,
  salvo la etapa opcional de energía cuando no hay fichero de energía.
- El análisis espectral no reutiliza la ventana lineal de un diagnóstico físico
  fallido. La reanudación mantiene la comprobación de código, entradas y opciones.
- Se respetan los CPU asignados por Slurm; las rutas con espacios funcionan.
- `FIELDS_GIF=0` desactiva realmente los GIF de campos.
- El control inicial rechaza valores no finitos, especies o datos ausentes y
  snapshots evolucionados usados como si fueran el estado inicial.
- La selección de corridas usa `analysis_config.json` cuando existe y exige el
  último snapshot programado según nmax y cadencia. Rechaza casos duplicados.
- Los errores del runner, comparaciones y análisis resonante hacen terminar el
  trabajo Slurm con error. Los estados científicos siguen siendo independientes.

## Validación local

- Batería completa: 299 pruebas y 14 subpruebas aprobadas antes de añadir la
  comprobación ADIOS2 real; las 26 regresiones de `test_batch_reliability.py`,
  incluida esa comprobación adicional, también aprobaron.
- Archivos BP reales escritos con ADIOS2 2.12.1.1001: campos, momentos de
  partícula y límites de ventana coinciden con su representación HDF5.
- Cuatro ejecuciones sintéticas del runner, cada una con sus 14 etapas por
  defecto más `theory`: mirror K3, firehose Maxwell, whistler Maxwell y M_S_bM.
  Las 60 etapas terminaron con PASS de ejecución.
- Etapas opcionales `theory-liouville` y `energy-if-present` ejecutadas aparte;
  la segunda sin archivo externo de energía, para comprobar su omisión válida.
- Reanudación real de M_S_bM: las 15 etapas se reconocieron como actualizadas.
  Una reanudación con código modificado fue rechazada, como corresponde.
- Pruebas del script de lote con Slurm simulado: un fallo del runner o del
  análisis resonante se propaga al código de salida del trabajo.

Los datos sintéticos usaron malla 32, 21 snapshots de campos, 400 partículas
por celda en el generador, dos etapas concurrentes y muestreo reducido de mapas,
VDF y teoría. Los tiempos de 41–52 segundos por perfil **no estiman** el coste
de las corridas reales. PASS de ejecución no certifica la interpretación física;
hay que consultar los estados científicos y el informe de calidad.

## Ejecución en COSMA

Después de llevar estos cambios al checkout de COSMA, usar una carpeta nueva
para no mezclar resultados calculados con versiones anteriores:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
export NEW_ROOT="$PWD/analysis_results/v6-reviewed-20261014"
export FIELDS_GIF=0
sbatch --export=ALL cosma_jobs/analisis/reanalysis_v6_all.sh
```

El script incluye cinco corridas concretas: mirror Maxwell/K3/K5 y firehose
Maxwell/K3. Otras corridas, incluidos whistler y controles, se añaden antes de
`sbatch` con `EXTRA_RUNS='PERFIL:carpeta PERFIL:carpeta'`; las carpetas se buscan
en RUN_ROOT. No descubre automáticamente todas las simulaciones del disco.
Consultar `RUNS` en el script para comprobar los identificadores exactos.

Revisar las directivas `#SBATCH` y ajustar `--time`/`--nodes` al presupuesto real
antes de enviar: el valor existente de tiempo es 48 horas. No se ha enviado
ningún trabajo desde esta revisión. Se conserva el preflight de pruebas.

Para continuar un trabajo interrumpido, conservar NEW_ROOT y todas las opciones,
exportar `RESUME=1` y volver a enviar. Si cambian código, entradas u opciones,
usar otra carpeta; el runner rechaza mezclar versiones. Consultar
`REANALYSIS_SUMMARY_<jobid>.txt`, `quality_report/index.html` y los `pipeline.json`
antes de utilizar las figuras. Las corridas incompletas o ausentes se enumeran
como SKIP: un trabajo que termina no implica que se hayan analizado esas corridas.

`PHYSICS_FLAGS` admite opciones adicionales al ejecutar Make o el runner
directamente, por ejemplo `--make-option 'PHYSICS_FLAGS=--jobs 4'`. Reducir
muestreo o bootstrap cambia la cobertura; no se recomienda hacerlo sin valorar
qué medidas necesita el análisis final.
