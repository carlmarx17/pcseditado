# Estado de la pipeline de análisis

Este archivo registra decisiones de mantenimiento actuales. La guía de uso está
en `CodeforAnalisys/README.md`.

## Decisiones actuales

- `psc_units.py` es la fuente compartida para perfiles físicos, normalización,
  patrones de archivo y nombres de partículas.
- `anisotropy_analysis.py` calcula anisotropía y beta usando presión térmica
  central proyectada respecto al campo local.
- `heat_flux_analysis.py` puede correr sin SciPy si `sigma=0`; con suavizado
  (`sigma > 0`) requiere `scipy.ndimage`.
- `spectral_analysis.py` conserva `--outdir` para seleccionar la carpeta de
  salida de espectros.
- Los perfiles `*_lite` no forman parte del flujo de producción.
- Los scripts cualitativos o de otra física se mantienen en `legacy/` y no se
  exponen como targets de `Makefile`.
- `physical_diagnostics.py` no genera figuras que solo indican disponibilidad
  de snapshots ni curvas de heat flux etiquetadas como proxy de partículas; las
  VDF individuales y las tablas CSV son la salida verificable.
- La figura de error de energía solo se genera cuando el paso contiene energía
  cinética de partículas y energía magnética finitas; no se mezclan pasos
  incompletos.

## Alcance mantenido

Incluido:

- análisis de anisotropía, campos, partículas, espectros, corrientes
  diamagnéticas y flujo de calor;
- generación de manifiesto por corrida;
- compatibilidad con los casos `M_*_bM`, `F_*_bM`, `W_*_bM` y perfiles
  Kappa/Maxwellian mantenidos.

Excluido:

- figuras generadas;
- `__pycache__`;
- notebooks o presentaciones;
- scripts experimentales no conectados al `Makefile`.
- figuras cualitativas archivadas en `legacy/`;
- proxies no defendibles como figuras principales de tesis.
