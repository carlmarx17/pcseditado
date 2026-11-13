# Análisis pendientes en COSMA

Desde el repositorio actualizado en COSMA, el primer comando muestra las historias disponibles y el segundo envía la tanda.

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
git pull --ff-only origin main
bash cosma_jobs/analisis/tesis_pendientes.sh
bash cosma_jobs/analisis/tesis_pendientes.sh --submit
```

El plan descubre mirror, firehose y whistler en `RUN_ROOT` (por defecto `anisotropy_adios2`). Mantiene por separado las repeticiones, los controles y las variantes de resolución aunque compartan perfil. Incluye historias con evolución disponible, también las parciales de cualquier familia. El plan y `revision/history_coverage.json` guardan el último paso disponible, el esperado y el indicador de cobertura parcial. El manifiesto comprueba la condición inicial y la geometría. El tiempo final por sí solo no garantiza cobertura continua. Las carpetas sin campos, las que sólo contienen la condición inicial y las familias ausentes aparecen en la consola.

Se usa una tarea de una CPU, con una historia activa a la vez y un límite de 48 horas por historia. El tiempo real depende de las instantáneas, el tamaño de malla, los productos reutilizados y el sistema de archivos. Las etapas que se interrumpen se reinician por separado. Los archivos `pipeline.json` guardan segundos y pico de memoria por etapa, en KiB en Linux. Las etapas con `archive` conservan los recursos históricos. Para consultar lo que contabilizó Slurm, usar el identificador impreso al enviar.

```bash
sacct -j ID --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS
```

El destino estable es `analysis_results/tesis_pendientes`. Cada envío lleva un plan y un log por tarea. Al reenviar, las etapas completas conservan sus productos. Para los archivos anteriores en `ARCHIVES` (por defecto `analysis_results`), se comprueban la historia de entrada, configuración, opciones, código de cálculo y archivos de salida registrados. Se reutilizan mediante enlaces, por lo que los árboles originales deben seguir disponibles. Una copia con fechas modificadas, un resultado suelto o una etapa antigua de procedencia insuficiente pueden requerir cálculo. La unidad de reutilización es la etapa íntegra. Si cambia el código o crece la historia, elegir otro `OUT` bajo `ARCHIVES` para recuperar las etapas compatibles.

## Qué revisar

* Abrir `revision/energy_audit.csv` para la auditoría conjunta, controles compatibles y cobertura. El informe se crea cuando terminan las historias seleccionadas. Cada historia tiene además `report/index.html`, su registro de ejecución y diagnósticos por especie.
* En `runs/<historia>/<perfil>/11_heating/`, revisar `heating_channels.png`, su tabla y `budget_summary.json`. El cierre global usa las seis componentes de los campos y las cinéticas relativistas de iones y electrones de DiagEnergies. El resumen conserva la deriva máxima, la final y los cambios por fase cuando hay un ajuste aceptado. Si faltan los diagnósticos globales, la reconstrucción por momentos mantiene su carácter aproximado.
* Comparar por especie el cambio cinético con la integral de la corriente por el campo eléctrico y la energía interna con la integral de menos presión por gradiente de velocidad. Cada integral mide una transferencia ya incluida en las reservas. La cadencia, el desfase de malla y la aproximación de momentos limitan su precisión. La deriva global cuantifica energía neta creada o perdida. Para estimar la parte numérica de la energización electrónica también importan el control y la convergencia. Un control isotrópico mide el fondo bajo las condiciones de esa corrida. Verificar solver, forma de partícula y parámetros efectivos antes de usar la resta en el artículo.
* La comparación inicial a κ variable fija el segundo momento central de la distribución cargada. Las columnas históricas `T_parallel` y `T_perp` de partículas representan componentes de la covarianza central ponderada de u y v multiplicada por la masa. Los mapas calculan P/n tras restar el flujo por celda, con aproximación no relativista en el término de deriva. La partición interna por momentos y la energía cinética relativista tienen estimadores distintos.
* En mirror revisar junto al crecimiento espectral `oblique_phase.csv` y `deepest_holes.csv` cuando haya mediciones suficientes. El trabajo conserva los armónicos de la caja, la frecuencia, la polarización y la compresibilidad. En firehose y whistler incluye el espectro, polarización, teoría paralela y anisotropía en bandas de velocidad para la especie impulsora. La identificación de una rama requiere que k y ω sean resolubles.

## Reconexión

La doble hoja de Harris usa otra geometría y sus propios perfiles. Para una historia conocida, enviar el trabajo separado desde el repositorio. Sustituir la ruta y seleccionar el perfil del ejecutable histórico (`reconnection` o `reconnection_comparable`).

```bash
sbatch cosma_jobs/analisis/tesis_reconexion.sh /ruta/a/la/historia reconnection_comparable
```

El trabajo escribe la evolución de flujo y κ para las salidas disponibles, y un balance global con la frecuencia ciclotrón del perfil de reconexión. Ambos ejecutables mantenidos tienen contornos periódicos. DiagEnergies agrega por signo de carga las partículas de la hoja y del fondo. El perfil histórico puede carecer de ese diagnóstico. Se registra esa ausencia. El trabajo detecta resultados completos de este envío en el destino estable. También busca resúmenes históricos en `ARCHIVES` y enlaza los productos compatibles con la ruta, geometría, serie, cobertura y fechas de las entradas. El registro guarda el origen. Los resúmenes históricos de reconexión carecen de huella de código, y esa reutilización ofrece trazabilidad más limitada.


## Añadir firehose tras el envío 12350652

La primera versión del plan excluyó las historias de producción cortas. La versión corregida acepta su evolución disponible. Para terminar el envío activo con la misma revisión de código, esperar a que finalicen el array 12350652 y la revisión 12350653 antes de actualizar el checkout. Después enviar únicamente firehose a su propio destino.

```bash
git pull --ff-only origin main
OUT="$PWD/analysis_results/tesis_firehose_disponible" bash cosma_jobs/analisis/tesis_pendientes.sh --family firehose
OUT="$PWD/analysis_results/tesis_firehose_disponible" bash cosma_jobs/analisis/tesis_pendientes.sh --family firehose --submit
```

Los resultados parciales permiten medir la evolución observada y ajustar la fase lineal cuando esté cubierta. Las inferencias sobre saturación y estado tardío deben declarar el intervalo observado.
