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
