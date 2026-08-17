# Analysis pipeline status

This file records current maintenance decisions. The usage guide is in
`CodeforAnalisys/README.md`.

## Current decisions

- `psc_units.py` is the shared source for physical profiles, normalization, file
  patterns and particle names.
- `anisotropy_analysis.py` computes anisotropy and beta using the central thermal
  pressure projected onto the local field.
- `heat_flux_analysis.py` can run without SciPy if `sigma=0`; with smoothing
  (`sigma > 0`) it requires `scipy.ndimage`.
- `spectral_analysis.py` keeps `--outdir` for selecting the spectra output
  folder.
- The `*_lite` profiles are not part of the production workflow.
- Qualitative scripts, or scripts belonging to other physics, are kept in
  `legacy/` and are not exposed as `Makefile` targets.
- `physical_diagnostics.py` does not generate figures that only indicate snapshot
  availability, nor heat-flux curves labelled as particle proxies; the individual
  VDFs and the CSV tables are the verifiable output.
- The energy-error figure is only generated when the step contains finite
  particle kinetic energy and magnetic energy; incomplete steps are not mixed in.

## Maintained scope

Included:

- analysis of anisotropy, fields, particles, spectra, diamagnetic currents and
  heat flux;
- per-run manifest generation;
- support for the `M_*_bM`, `F_*_bM`, `W_*_bM` cases and the maintained
  Kappa/Maxwellian profiles.

Excluded:

- generated figures;
- `__pycache__`;
- notebooks and presentations;
- experimental scripts not wired into the `Makefile`;
- qualitative figures archived in `legacy/`;
- proxies that cannot be defended as main thesis figures.
