# PSC — Temperature-anisotropy instabilities

Fork of [PSC](https://github.com/psc-code/psc) (Particle-in-Cell, Kai Germaschewski
et al.) used as the numerical basis of a Physics MSc thesis at Universidad Nacional
de Colombia.

## Goal

Determine how the development of temperature-anisotropy instabilities in a
collisionless plasma changes when the velocity distribution is not Maxwellian.

Concretely, the three instability families that regulate anisotropy in space
plasmas — **mirror**, **firehose** and **whistler** — are simulated with kinetic
PIC, starting from two initial distributions with the *same* macroscopic
parameters (β, A = T⊥/T∥, mi/me):

- **bi-Maxwellian**, the reference case;
- **bi-Kappa** (κ = 3, 5), with suprathermal tails.

The question is what shifts when the distribution changes: the linear growth rate
γ, the spectrum of excited modes k(γ_max), the saturation level of δB, and the
final state the plasma relaxes to in the (β∥, A) plane — the "Brazil plot". The
repository contains both the simulation cases and the analysis pipeline that
produces those diagnostics.

## What this fork adds on top of PSC

The PSC core (field solver, particle pusher, I/O, load balancing) is **upstream and
unmodified**. What is specific to this repository:

| Component | Where | What it provides |
|---|---|---|
| Parameterized base case | `src/psc_anisotropy_case.hxx` | Template shared by all anisotropy cases: geometry, outputs, intervals and environment overrides. |
| bi-Kappa distribution | `src/psc_anisotropy_case.hxx` (`PSC_USE_KAPPA`, `PSC_KAPPA`) | Sampling of an anisotropic bi-Kappa as initial condition, an alternative to PSC's bi-Maxwellian. |
| 15 instability cases | `src/psc_{mirror,firehose,whistler}_*.cxx` | One executable per physical regime; each file only sets the label, distribution and parameters. |
| ADIOS2 checkpoint/restart | `src/cosma_*.sh`, `adios2cfg.xml` | Resumable long runs (BP5), required to reach saturation on HPC. |
| Analysis pipeline | `CodeforAnalisys/` | Full post-processing: anisotropy, spectra, dispersion, growth rates, physical diagnostics. |
| Cluster scripts | `cosma_jobs/` | SLURM simulation and analysis jobs, named after what they run. |

## Repository layout

```text
src/                    PSC code + anisotropy cases (.cxx) and physics docs
CodeforAnalisys/        Analysis pipeline (Python + Makefile)
CodeforAnalisysLocal/   Variant for small local runs
cosma_jobs/             SLURM jobs: simulacion/ and analisis/
presentation/           Presentation material
docs/, doxygen/         Upstream PSC documentation
python/, matlab/, ...   Upstream utilities
```

Analysis output (`analysis_results/`) is **not** tracked in git — it is regenerated
from the raw run data with the `CodeforAnalisys` Makefile targets. See
[Data policy](#data-policy).

## Simulation cases

Common configuration: `PscConfig1vbecSingle<dim_yz>`, domain 20 d_i × 20 d_i,
grid 576×576 (≈28.8 cells/d_i), mi/me = 200, vA/c = 0.08, periodic boundaries,
background field along **z** (T∥ = T_z, T⊥ = (T_x+T_y)/2).

### Mirror — ions with perpendicular excess
Criterion: `β_i∥ (A_i − 1) > 1`

| Executable | Regime | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_mirror_bimaxwellian_strong` | Strong | 5.0 | 3.0 | 1.0 | 1.0 |
| `psc_mirror_bimaxwellian_moderate` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 |
| `psc_mirror_bimaxwellian_weak` | Weak | 6.0 | 1.5 | 1.0 | 1.0 |

### Firehose — ions with parallel excess
Criterion: `β_i∥ (1 − A_i) > 2`

| Executable | Regime | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_firehose_bimaxwellian_strong` | Strong | 10.0 | 0.1 | 1.0 | 1.0 |
| `psc_firehose_bimaxwellian_moderate` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 |
| `psc_firehose_bimaxwellian_weak` | Weak | 3.0 | 0.6 | 1.0 | 1.0 |

### Whistler — electrons with perpendicular excess
Criterion: `A_e > 1 + 0.21 / β_e∥^0.6`

| Executable | Regime | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_whistler_bimaxwellian_strong` | Strong | 1.0 | 1.0 | 0.5 | 3.0 |
| `psc_whistler_bimaxwellian_moderate` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 |
| `psc_whistler_bimaxwellian_weak` | Weak | 1.0 | 1.0 | 0.5 | 1.5 |

### Bi-Kappa — non-Maxwellian counterparts

| Executable | κ | β_i∥ | A_i | Compares against |
|---|---:|---:|---:|---|
| `psc_mirror_bikappa3` | 3 | 5.0 | 3.0 | `psc_mirror_bimaxwellian_strong` |
| `psc_mirror_bikappa5` | 5 | 5.0 | 3.0 | `psc_mirror_bimaxwellian_strong` |
| `psc_mirror_bikappa3_moderate` | 3 | 5.0 | 2.0 | `psc_mirror_bimaxwellian_moderate` |
| `psc_firehose_bikappa3` | 3 | 10.0 | 0.1 | `psc_firehose_bimaxwellian_strong` |
| `psc_firehose_bikappa5` | 5 | 10.0 | 0.1 | `psc_firehose_bimaxwellian_strong` |

### Big box (40 d_i)

`psc_firehose_bimaxwellian_moderate_bigbox40` and `psc_firehose_bikappa3_bigbox40`
double the domain to accommodate the longer-wavelength firehose modes. Box size is
a **compile-time** setting (`PSC_DOMAIN_DI` in `psc_anisotropy_case.hxx`), which is
why these are separate executables.

Besides the anisotropy line of work, the repository keeps **magnetic reconnection**
cases (`psc_reconnection*.cxx`), documented in `src/SIMULACIONES_RECONNECTION.md`.

## Analysis pipeline (`CodeforAnalisys/`)

Takes the snapshots of a run (HDF5 `.h5` or ADIOS2 `.bp`) and produces the thesis
diagnostics. Orchestrated by `Makefile`; `PSC_PROFILE` sets the physical parameters
and the normalization of each case (`psc_units.py`).

```bash
cd CodeforAnalisys
make show-inputs DATA_DIR=/path/to/run CASE=mirror_bikappa3_moderate
make analysis    DATA_DIR=/path/to/run CASE=mirror_bikappa3_moderate
```

`make analysis` = `manifest` + the 8 stages of `common`, which can also be run
individually:

| Stage | Script | Produces |
|---|---|---|
| `brazil` | `anisotropy_analysis.py` | Evolution of A(t), β∥(t) and trajectory in the Brazil plot |
| `fields` | `fluctuationofmagneticfiel.py` | δB maps per snapshot and GIFs of the evolution |
| `spectral` | `spectral_analysis.py` | E(k, Ω_ci t), log-linear γ(k) per shell, helicity σ_m(k), compressibility |
| `dispersion` | `dispersion_analysis.py` | Dispersion relation ω(k) and modal density map ω/Ω_ci vs \|v_ph\|/v_A |
| `growth-map` | `growth_rate_map.py` | γ(k∥, k⊥) without radial binning — separates parallel from oblique modes |
| `polarization` | `polarization_dispersion.py` | Polarization branches ± and rates gated by R² |
| `diamagnetic` | `diamagnetic_current.py` | Diamagnetic current (mirror signature) |
| `heatflux` | `heat_flux_analysis.py` | Parallel and perpendicular heat flux |
| `particles` | `plot_prt.py` | Velocity distributions from the particle output |
| `validate` | `validate_moments.py` | Consistency of grid moments vs particle moments |
| `physics` | `physical_diagnostics.py` | Summary tables: energy, anisotropy, fits, correlations |

Extras: `compare_physical_cases.py` (bi-Maxwellian vs bi-Kappa comparison),
`check_resolution.py` (checks that the profile resolution matches the data), and
tests with `pytest`.

Each run writes `analysis_results/<CASE>/` with the structure `01_anisotropy/`,
`02_fields/`, `04_spectra/`, `05_diamagnetic/`, `06_heat_flux/`,
`09_physical_diagnostics/` and a `<CASE>_analysis_manifest.json` recording the
parameters and the detected steps.

## Data policy

Only source code, job scripts and documentation are tracked in git. Everything
that can be regenerated stays out:

- `analysis_results/`, `CodeforAnalisys/downloads/` — figures, GIFs and per-step
  maps. Rebuild with the Makefile targets above.
- `graphify-out/` — generated knowledge graph (machine-specific paths, AST cache).
- Raw run data (`*.bp`, `*.h5`, checkpoints) — lives on COSMA, never in the repo.

## Local build

```bash
cmake --build build --target psc_mirror_bimaxwellian_strong
```

```bash
cmake --build build --target psc_mirror_bikappa3
```

## Documentation

| Document | Contents |
|---|---|
| `src/SIMULACIONES_ANISOTROPIA.md` | Full catalogue of cases and their physical parameters |
| `src/SIMULACIONES_INESTABILIDADES_TEMPERATURA_ANALISIS.md` | Physical criteria and analysis plan for mirror / firehose / whistler |
| `src/ESCALADO_INESTABILIDADES.md` | Scaling and sizing of the runs |
| `src/REFACTOR_KAPPA.md` | Notes on the bi-Kappa distribution refactor |
| `src/SIMULACIONES_RECONNECTION.md` | Magnetic reconnection cases |
| `src/ADIOS2_COSMA_RUNBOOK.md` | ADIOS2/COSMA procedure (technical reference) |
| `CodeforAnalisys/README.md` | Day-to-day use of the pipeline |
| `CodeforAnalisys/ANALISIS_ESTRUCTURA.md` | File, reader and output contract |
| `cosma_jobs/README.md` | What each SLURM job does |

> Cluster operations (access, job submission, monitoring, checkpoints, common
> problems) are kept outside the repository, in Notion:
> **COSMA7 — Operational runbook**.

## License and credits

PSC is the work of Kai Germaschewski and collaborators (UNH, LMU); see `LICENSE`
and the git history. This fork adds only the anisotropy cases and the analysis
pipeline described above.
