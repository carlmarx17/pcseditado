# CodeforAnalisys Refactor Notes

## What Changed

- `aniso.py` was removed after its functionality was absorbed by `anisotropy_analysis.py`.
- `plotmirro.py` was removed after its functionality was absorbed by `mirror_physics.py`.
- `diamagnetic_current.py` was restored as a maintained standalone diagnostic for ion/electron/total diamagnetic-current maps.
- `plot_moments_validation.py` was removed after splitting validation and visualization responsibilities across `validate_moments.py`, `plot_prt.py`, `plot_vdf_3d.py`, and `plot_moments_scatter_3d.py`.
- The `Makefile` now runs the analysis scripts against `../build/src` data products and includes particle diagnostics plus `make diamagnetic`.
- `psc_units.py` now acts as the shared reference for unit conversions, common file patterns, and derived plasma scales.

## Duplicate Analysis

- Exact duplication detected:
  `plotmirro.py` at the repository root is effectively the same old implementation as the deleted `CodeforAnalisys/plotmirro.py`. It should stay out of the maintained analysis workflow.
- Functional replacement:
  `aniso.py` and `CodeforAnalisys/plotmirro.py` are legacy predecessors, not independent tools anymore.
- Restored but distinct:
  `diamagnetic_current.py` is no longer a stale note in the docs; it is back as a real maintained diagnostic with its own outputs.
- Shared input, different outputs:
  `plot_prt.py`, `plot_vdf_3d.py`, and `plot_moments_scatter_3d.py` all consume `prt.*.h5`, but they produce different views of the particle distribution.
- Shared field source, different analyses:
  `fluctuationofmagneticfiel.py`, `mirror_physics.py`, `diamagnetic_current.py`, and `spectral_analysis.py` consume overlapping PSC dumps, but they are not interchangeable.

## C++ Side

- `SetupParticles` gained `createKappaMultivariate()` and a configurable `kappa` parameter.
- `psc_temp_aniso.cxx` now initializes particles with the multivariate Kappa sampler and re-enables particle output every 100 steps.
- Default field writing was forced to `WriterMRC`, removing the ADIOS2 conditional in the current build path.
- The grid decomposition was adjusted from 48 patches to 4 patches.

## Commit Scope

- Included:
  tracked C++ changes, tracked analysis refactors, new maintained analysis scripts, and documentation updates.
- Excluded:
  generated plots, `__pycache__`, editor settings, PDFs, and the root-level legacy `plotmirro.py`.
