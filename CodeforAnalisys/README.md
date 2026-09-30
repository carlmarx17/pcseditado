# PSC output analysis

This folder contains the maintained pipeline used to analyse the anisotropy runs
`M_*_bM`, `F_*_bM`, `W_*_bM` and the Kappa/Maxwellian cases.

Physical audit and publication priorities (2026-09-09):
[AUDITORIA_FISICA_PAPER.md](AUDITORIA_FISICA_PAPER.md).
Regenerate affected outputs in a new results directory after this revision.
The Alfvén speed now satisfies `VA = OMEGA_CI * DI`; the historical profile
key `vA_over_c` is the simulated `B0`, not the physical Alfvén speed.
The energy table now reports `E_proxy` and `energy_proxy_relative_change`,
not a conserved total. `electron_energy_trend.csv` describes a fit without
subtracting it or attributing it to numerical heating. Old corrected-energy
files in existing output directories are obsolete and are not removed automatically.

**Revision 2026-09-28 (analysis conventions version 5).** Every product made
before this date must be regenerated; the manifest records the version.

- gamma is fitted on the vector fluctuation \(\langle|\delta\mathbf B|^2\rangle^{1/2}\),
  not on \(|B|-B_0\): for transverse modes (parallel firehose, EMIC, whistler)
  \(|B|-B_0\sim\delta B_\perp^2/2B_0\) is second order and gave **2 gamma**.
- One linear-phase fit for the whole pipeline (`growth_fit.py`) with
  `gamma_err` = slope error (+) window sensitivity; the spectral fits used a
  slope-sign window that biased gamma low.
- The diamagnetic current of `physical_diagnostics.py` had the opposite sign
  to `diamagnetic_current.py`, used the raw second moment and per-cell
  gradients; both now use the thermal \(P_\perp\) on the local field, gradients
  in \(d_e\), and \(\mathbf J=\mathbf B\times\nabla P_\perp/B^2\).
- `heat_flux_analysis.py` measured \(P_\parallel U_\parallel\), which is not a
  heat flux; it now measures the third central moment from the particles
  (local frame, truncation for kappa = 3, sampling error and noise floor).
- Whistler (electron-driven) cases: the spectral analysis and the growth map
  cut at \(k d_i = 2\) and 1.5, below the whole whistler band
  (\(k d_i \sim 3-14\)); maps, Brazil plot and comparisons were for ions only;
  the theory had no whistler branch and its overlay was dropped. All fixed.
- Particle temperatures use PSC's \(\langle u\,v\rangle\) pressure convention
  (the one of the moment maps), and energies the exact \(m(\gamma-1)\).
- New analyses: field-solver residuals, saturated structures and pressure
  balance, \(J_s\cdot E\) energy exchange, estimator comparison and
  convergence/realization study (see "Thesis workflow" below).

**Follow-up (after the first v5 reanalysis, job 12069106).** gamma changes
again; regenerate `09_physical_diagnostics` and `04_spectra`.

- The reference gamma of a run is now the fit of its **dominant Fourier mode**
  of \(\delta\mathbf B\) (`growth_rate_summary.csv`, series `mode`;
  `mode_growth_table.csv`; `growth_rate_fit_mode.png`). The domain rms mixes
  the noise of every k with the mode and its log-slope was ~half the mode's
  (mirror moderate: 0.056-0.065 vs ~0.11-0.12). The rms series (`total`,
  `parallel`, `perp`) are fitted on the mode's linear phase. Comparisons,
  convergence study and `kappa_evolution.py` all read the `mode` row
  (`growth_fit.reference_growth_row`).
- **Branch-resolved growth rates (v6 follow-up 4).** At beta_i|| = 5, A_i = 2
  the mirror and ion-cyclotron (IC) thresholds almost coincide, and the v5
  dominant mode was an IC wave (k_perp = 0, compressibility ~1e-12). Every
  followed mode is now classified by theta_kB and compressibility
  |dB_par|^2/|dB|^2 (`physical_diagnostics.classify_mode`): compressive
  oblique (theta_kB >= 45 deg, >= 0.5; mirror for T_perp > T_par ions) or
  transverse parallel (theta_kB <= 30 deg, <= 0.2; IC). The strongest mode of
  each branch is fitted in its own row, `mode_compressive` and
  `mode_transverse`, with `growth_rate_fit_mode_compressive.png` /
  `_transverse.png`; `linear_phase.json` carries the classification and a
  `branches` block, the quality report a `branch_growth` check, and
  `compare_physical_cases.py` the figure `comparison_growth_rate_branches.png`.
  The mirror growth rate of the thesis is the `mode_compressive` row.
- **psi_pm handedness is verified, not assumed.** `polarization_dispersion.py`
  runs `handedness_check` before every analysis: the ion gyration sense is
  integrated with the Boris rotation, waves rotating in that sense (left-hand)
  and in the opposite sense are pushed through the same transforms, and the run
  stops if they do not land on psi_+ / psi_- at omega > 0. psi_pm are now
  defined relative to B0 (sign of b0 included), titles state the handedness,
  and the JSON records the peak (k, omega) of each channel.
- **Publication figures from analysis products (`paper_figures.py`).** Built
  on a laptop from the downloaded CSV/JSON of a controlled series, no raw
  data needed: amplitude of the dominant ion-cyclotron mode with the linear-
  theory slope, gamma vs 1/kappa against the parallel kinetic dispersion
  relation, the (beta_i||, A_i) trajectories with the thresholds of the
  measured electrons, the reduced distributions at the cyclotron resonance,
  and the local-frame 1/kappa_eff (the global-B0 fit of `kappa_evolution.py`
  reads the wave-tilted distribution at saturation as a tail), plus
  `series_summary.csv` with the numbers quoted in the text:
  `python paper_figures.py RUN_MAXW RUN_K5 RUN_K3 --outdir OUT`.
- **Mirror threshold with the measured electrons.** The Brazil plots and the
  anisotropy evolution of mirror cases use the Hellinger (2007) criterion for
  bi-Maxwellian ions and electrons with beta_e||(t) and A_e(t) measured at each
  snapshot (`plasma_physics.mirror_threshold_electrons`), since the electrons
  heat from beta_e = 1 to ~8. The cold-electron curve is kept in the CSV.
- The automatic window no longer locks onto the PIC quiet-start build-up of
  the noise floor (the first ~2 \(\Omega_{ci}^{-1}\), steeper than any
  instability): the spectral, polarization and dispersion fits of the v5 run
  sat at \(t<7\,\Omega_{ci}^{-1}\), one with \(\gamma\approx0.6\) flagged valid.
- `growth_rate_map.py` fits every mode on the linear phase of its dominant mode
  instead of 10-60 % of the run, which reached into saturation
  (`fit_window_source` in the JSON; `--fit-lo/--fit-hi` keep the old choice).
- `compare_physical_cases.py` drew empty anisotropy and heat-flux panels: those
  columns only exist at the particle cadence and the line broke at every NaN.

## Thesis workflow

One command produces every product of one run, in dependency order
(initial-condition check and manifest, validation evidence, then physics):

```bash
make thesis DATA_DIR=/path/to/run CASE=mirror_bimaxwellian_moderate \
    [RUN_TAG=seed1] [LOGS=/path/to/job.out] [GROWTH_T_START=5 GROWTH_T_END=20]
```

`GROWTH_T_START/END` fix the linear phase (in \(\Omega_{ci}t\)) of every gamma
fit once it has been read off `growth_rate_fit.png` and the growth-rate map;
without them the automatic window is used and its sensitivity is part of
`gamma_err`. `LOGS` defaults to the COSMA job log next to the run directory.

Before spending COSMA time, validate the whole chain on a synthetic run with
known answers (prescribed gamma = 0.25 Omega_ci, divergence-free Yee field,
pressure-balanced structures, bi-Maxwellian particles, conserved energy):

```bash
python synthetic_run.py /tmp/syn --case mirror_bimaxwellian_moderate
make thesis DATA_DIR=/tmp/syn RESULTS_ROOT=/tmp/syn_results
```

`test_pipeline_synthetic.py` runs the same check inside the test suite.

| Thesis figure (audit proposal) | Evidence | Target | Main files |
|---|---|---|---|
| 1. Initial VDFs and measured parameters | measured n, B0, T, A, beta vs declared | `manifest`, `validate`, `particles` | `check_initial_conditions` output, `08_validation/`, `03_particles/` |
| 2. Energy validation and convergence | global energy, div B, Gauss, continuity; dx/dt/ppc/seed spread | `energy`, `residuals`, `convergence` | `global_energy_*`, `field_residuals*`, `convergence_*` |
| 3. gamma(k_par, k_perp), mode identification, theory | growth map, polarization, helicity, compressibility, PIC vs theory | `spectral`, `theory`, `polarization` | `04_spectra/` |
| 4. A(t), beta(t), dB(t), saturation | trajectories of the driven species, gamma with error | `physics`, `brazil`, `estimators` | `09_physical_diagnostics/`, `01_anisotropy/` |
| 5. Structures | holes/peaks, |B| skewness, n-|B| correlation, pressure balance | `structures` | `07_structures/` |
| 6. Local VDF / kappa_eff / closures | spatial VDF, kappa_eff, Liouville closures | `vdf-spatial`, `theory-liouville*` | `03_particles/`, `04_spectra/` |
| Energy transfer (P1) | J_s·E per species and channel vs DiagEnergies | `energy-exchange` | `energy_exchange_*` |
| Heat flux (P2) | third moment, truncation, noise floor | `heatflux` | `06_heat_flux/` |

## Expected input

Each data directory must contain a single PSC run:

```text
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_<case>.<step>.h5
```

ADIOS2 checkpoints (`checkpoint_<step>.bp/`) are for restarting the simulation.
The analysis pipeline works on the HDF5 field, moment and particle outputs.

## Quick start

From `CodeforAnalisys`:

```bash
make show-inputs DATA_DIR=/path/to/run CASE=M_S_bM
```

```bash
make analysis DATA_DIR=/path/to/run CASE=M_S_bM
```

It can also be run per case:

```bash
make F_M_bM DATA_DIR=/path/to/F_M_bM
```

To run the spectral analysis only:

```bash
make spectral DATA_DIR=/path/to/run CASE=F_S_bM_local
```

The script automatically detects the non-degenerate physical plane (`xy`, `xz`
or `yz`) and takes the cell spacing in units of \(d_i\) from the profile selected
via `PSC_PROFILE`. Instead of a static 1D/2D spectrum per snapshot,
`spectral_analysis.py` accumulates \(E(k,\Omega_{ci}t)\) over all snapshots and
fits a log-linear \(\gamma(k)\) for each \(k\) shell
(`growth_rate_by_k_<plane>.csv`, `growth_rate_vs_k_<plane>.png`,
`energy_kt_{perp,parallel}_<plane>.png`), plus the reduced magnetic helicity
\(\sigma_m(k)\) and the compressibility \(\delta B_\parallel^2/(\delta
B_\parallel^2+\delta B_\perp^2)\) — the mirror / EMIC / firehose discrimination
technique of Block 1.3-1.4. The same target also generates
`dispersion_density_<plane>_perp_absolute.png`, a modal density map of
\(\omega/\Omega_{ci}\) against \(|v_{\rm ph}|/v_A\), overlaying the
highest-power ridges as black points.

The same target additionally produces `growth_rate_map_<plane>_<component>.png`
and `.csv`, a direct \(\gamma(k_\parallel,k_\perp)\) map without radial binning.
This diagnostic preserves the mode geometry: peaks on the \(k_\parallel\) axis
indicate parallel modes, while off-axis peaks identify oblique modes such as
mirror or oblique firehose.

The temporal FFT is zero-padded to draw the ridges continuously; this
interpolates the spectrum, but does not increase the number of physically
independent frequencies, which is set by the number and cadence of snapshots.

To generate only that diagram:

```bash
make dispersion DATA_DIR=/path/to/run CASE=F_S_bM_local
```

### Detecting and characterizing the dominant magnetic mode

`make dispersion` now analyses all three magnetic components by default
(`DISPERSION_COMPONENT=total`). It writes `dispersion_modes_<plane>_total`
as PNG/PDF, JSON and CSV alongside the native omega-k map and the auxiliary
phase-velocity projection. Start with the **mode summary**, not the smoothed
velocity projection. Existing files with `_perp_` names are older or explicitly
transverse-only analyses; they are not replaced by the new `_total_` outputs.
`dispersion_modes_<plane>_total_fit.png`/PDF shows the measured amplitude and
unwrapped phase against the exponential and constant-frequency fits, including
their acceptance status. The JSON records the input files, profile and options.

The summary measures discrete `(k_parallel, k_perp)` pairs before perpendicular
reduction, display smoothing or de-growth. It reports the strongest spatial
peak, its share of retained time-averaged magnetic power, angle to the specified
background-field axis, signed frequency, growth fit, magnetic compressibility
and transverse polarization. Conjugate Fourier pairs are counted once. Power
ranking refers to the selected interval and retained k band; it is distinct
from ranking by growth rate. A smaller, coherent peak does not silently replace
an incoherent power maximum as the dominant mode.

Frequency uses `B = Re[b exp(i k.x - i omega*t)]`, with canonical
`k_parallel >= 0`; positive/negative omega means propagation along/against the
parallel axis. The native signed omega-k plot instead keeps omega >= 0 and
both signs of k. Frequencies are in the simulation frame; no flow/Doppler
correction is made. `sigma_b_transverse = 2 Im(b1 conj(b2))/(|b1|²+|b2|²)`
uses a right-handed basis about the parallel axis; its sign is reported without
assigning an EMIC/whistler/firehose species label. Such identification also
needs plasma parameters, a frame convention and comparison to theory.

The phase and log-amplitude fits assume a single complex exponential at each
spatial peak. Acceptance requires phase coherence >= 0.85, log-amplitude
residual standard deviation < 0.35, at least six contiguous nonzero snapshots
and separation from temporal Nyquist. These are diagnostic thresholds, not
statistical confidence levels. The growth fit is usable only when R² >= 0.8,
the interval spans >= 0.25 fitted e-foldings and the two half-interval slopes
agree within 50% of the full slope. Exact zero initial amplitudes are excluded
from logarithmic fits. Beating, growth followed by saturation, or a changing
phase can therefore remain unconfirmed even when magnetic power is strong.

`omega_resolution_over_omega_ci = 2*pi/T` comes from the actual usable time
span; padding the FFT does not improve it. Frequency below that resolution is
reported as unresolved, including aperiodic candidates. The default linear
frequency axis includes omega=0. The ridge CSV measures local frequency peaks
at native k bins without smoothing between wavenumbers; secondary peaks must
exceed the predicted taper sidelobe power by a factor of four. Peak ranks are local
to each k and do not establish branch identity. No-output/no-signal cases
produce an empty table and an explicit JSON status.

Use a time interval within one physical phase and a justified wavenumber band,
for example for the documented local run:

```bash
make dispersion DATA_DIR=../corridas_locales/mi_prueba CASE=F_S_bM_local \
  DISPERSION_COMPONENT=total DISPERSION_KMAX_DI=2 \
  DISPERSION_T_START=0.27 RESULTS_ROOT=../analysis_results/dispersion_review
```

`DISPERSION_KMAX_DI` caps the magnitude `sqrt(k_parallel²+k_perp²)*d_i`.
`DISPERSION_T_START` and `DISPERSION_T_END` are in `Omega_ci*t`, not steps.
An optional mode preset restricts the search to its angular/k band; it is a
prior choice, not evidence that the corresponding instability was detected.
Select `DISPERSION_MODE=generic` for an unrestricted angular search.

Synthetic regressions cover isolated waves, both propagation directions,
oblique and purely perpendicular growth, circular polarization, multiple
frequencies, noise, zero initial fluctuations, saturation and FFT padding:

```bash
python -m pytest -q test_dispersion_analysis.py test_dispersion_synthetic.py test_dispersion_modes.py
```

The transform convention follows the [NumPy DFT definition](https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details):
spatial `fft` is paired with temporal `ifft` (with normalization restored),
so a wave `cos(k.x-omega*t)` has its positive-frequency peak at positive k.

To generate only the \(\gamma(k_\parallel,k_\perp)\) map:

```bash
make growth-map DATA_DIR=/path/to/run CASE=M_M_bM
```

To run the same kind of diagnostic on the temperature anisotropy
\(A=T_\perp/T_\parallel\), reading moments and fields:

```bash
make anisotropy-dispersion DATA_DIR=/path/to/run CASE=M_M_bM
```

That target writes into `04_spectra/` a density map of \(\omega/\Omega_{ci}\)
against \(|v_{\rm ph}|/v_A\), a CSV with the dominant ridges, a time summary of
\(A\), and a linear k-space panel for the last analysed snapshot.

Regression tests are run with:

```bash
../.venv/bin/python -m unittest -v test_spectral_analysis.py test_dispersion_analysis.py
```

## Supported cases

| `CASE` | Instability | Species | Initial parameters |
|---|---|---|---|
| `M_S_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=3.0` |
| `M_M_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=2.0` |
| `M_W_bM` | Mirror | ion | `beta_i_parallel=6`, `A_i=1.5` |
| `F_S_bM` | Firehose | ion | `beta_i_parallel=10`, `A_i=0.1` |
| `F_M_bM` | Firehose | ion | `beta_i_parallel=6`, `A_i=0.3` |
| `F_W_bM` | Firehose | ion | `beta_i_parallel=3`, `A_i=0.6` |
| `W_S_bM` | Whistler | electron | `beta_e_parallel=0.5`, `A_e=3.0` |
| `W_M_bM` | Whistler | electron | `beta_e_parallel=0.5`, `A_e=2.0` |
| `W_W_bM` | Whistler | electron | `beta_e_parallel=0.5`, `A_e=1.5` |

The table above lists the legacy profiles. The maintained production cases
(576x576, 20 d_i, 1000 ppc; `*_bigbox40`: 1152x1152, 40 d_i) are:

| `CASE` family | Driven species | beta_par, A of the driven species | Distributions |
|---|---|---|---|
| `mirror_*_strong` / `_moderate` / `_weak` | ion | 5 / 3.0, 5 / 2.0, 6 / 1.5 | bi-Maxwellian; kappa 3, 5 (strong and moderate) |
| `firehose_*_strong` / `_moderate` / `_weak` | ion | 10 / 0.1, 6 / 0.3, 3 / 0.6 | bi-Maxwellian; kappa 3, 5 (strong) |
| `firehose_*_bigbox40` | ion | 10 / 0.1, 6 / 0.3 | bi-Maxwellian; kappa 3, 5 (strong) |
| `whistler_*_strong` / `_moderate` / `_weak` | electron | 0.5 / 3.0, 0.5 / 2.0, 0.5 / 1.5 | bi-Maxwellian; kappa 3 (moderate) |

The whistler moderate twins write fields every 100 steps and particles every
2000 (\(\Delta t\,\Omega_{ce}=2.64\), Nyquist \(1.19\,\Omega_{ce}\)); the profile
records it (`fields_every`, `particles_every`).

`psc_units.py` defines the physical profiles and output names. Do not use one
production profile to analyse a different case: `F_M_bM` is not equivalent to
`firehose_maxwellian`.

## Outputs

Output is written under:

```text
analysis_results/<CASE>/
```

Main subfolders:

```text
01_anisotropy/     evolution of A, beta and Brazil-plot trajectory
02_fields/         field and fluctuation maps
03_particles/      VDFs and particle moments
04_spectra/        mode-resolved gamma(k), E(k,t) map, helicity and compressibility
05_diamagnetic/    diamagnetic currents
06_heat_flux/      particle heat flux (third moment), truncation, noise floor
07_structures/     magnetic holes/peaks, |B| skewness, pressure balance
08_validation/     pointwise validation against particles
09_physical_diagnostics/ integrated diagnostics with the standard outputs
10_reconnection/   double-Harris diagnostics and the B–kappa correlation
```

Note that `analysis_results/` is **not** tracked in git — everything under it is
regenerated by these targets from the raw run data.

The integrated target:

```bash
make physics DATA_DIR=/path/to/run CASE=M_M_bM
```

writes into `09_physical_diagnostics/` the tables and figures of the physics
checklist: `validation_table.csv`, `validation_summary.txt`,
`anisotropy_table.csv`, `fit_metrics.csv`, `field_fluctuation_table.csv`,
`growth_rate_summary.csv`, `anisotropy_spatial_stats.csv`,
`spatial_correlations.csv`, `energy_table.csv`, plus `T_parallel/T_perp/A`
maps of the driven species, `deltaB`, `mirror_holes` and `J_dia` maps, 2D
VDFs, Maxwellian/Kappa fits, growth rate (the dominant Fourier mode, which is
the reference, and the total, parallel and perpendicular rms on its linear
phase, with `gamma_err`; every followed mode in `mode_growth_table.csv`),
correlations and energy. The same folder
receives `global_energy_*` (`make energy`), `field_residuals*`
(`make residuals`), `energy_exchange*` (`make energy-exchange`) and
`estimator_consistency*` (`make estimators`).

To compare already-analysed cases, for example Maxwellian vs Kappa:

```bash
make compare-physics COMPARE_CASES="maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics"
```

This produces `comparison_kappa_vs_maxwellian.csv`, `comparison_anisotropy.png`
(driven species), `comparison_deltaB.png`, `comparison_deltaB_components.png`,
`comparison_growth_rate.png` (with `gamma_err`), `comparison_energy.png`
(DiagEnergies) and `comparison_heat_flux.png` (third moment). The runs must
share every declared parameter except the distribution; cases driven by
different species are refused.

Numerical convergence and realizations (same case, different dx/dt/ppc/seed,
each analysed with its own `RUN_TAG`):

```bash
make convergence CONVERGENCE_RUNS="base=../analysis_results/runA dx2=../analysis_results/runB seed2=../analysis_results/runC"
```

writes `convergence_table.csv` (each observable, its change with respect to
the reference and which parameter changed), `convergence_groups.csv` (mean
and spread over realizations) and `convergence.png`.

## Physical background, formulas and variables

This section explains which physical question each analysis answers. The formulas
are written in PSC normalized units, where \(\mu_0=1\), \(c=1\), \(n_0=1\),
\(m_e=1\) and \(m_i/m_e=200\).

### 1. Building the thermal pressure

#### 1.1. Physical rationale

The moment files contain full second moments, which mix thermal motion and
collective plasma motion. To measure temperature, anisotropy or beta, the
contribution of the macroscopic velocity must be removed first. Otherwise a
plasma flow could be incorrectly interpreted as heating.

#### 1.2. Formulas

For each species \(s\):

$$
n_s = |\rho_s|,
\qquad
u_{i,s} = \frac{p_{i,s}}{n_s m_s},
$$

$$
P_{ij,s}
= M_{ij,s}
- \frac{p_{i,s}p_{j,s}}{n_s m_s}
= M_{ij,s}-n_s m_s u_{i,s}u_{j,s}.
$$

#### 1.3. Variables

1. \(s\): species, ion \(i\) or electron \(e\).
2. \(n_s\): number density of the species.
3. \(\rho_s\): density as stored by PSC; for electrons its magnitude is used.
4. \(m_s\): species mass.
5. \(p_{i,s}\): first momentum moment along direction \(i\).
6. \(u_{i,s}\): macroscopic or drift velocity.
7. \(M_{ij,s}\): raw second moment stored as `txx`, `txy`, etc.
8. \(P_{ij,s}\): central thermal pressure tensor.

### 2. Pressure and temperature relative to the magnetic field

#### 2.1. Physical rationale

The Mirror, Firehose and Whistler instabilities depend on the difference between
the pressure parallel and perpendicular to the magnetic field. The physically
relevant direction is the local field, not necessarily a fixed grid axis.

#### 2.2. Formulas

$$
\mathbf{B}=(B_x,B_y,B_z),
\qquad
B=|\mathbf{B}|=\sqrt{B_x^2+B_y^2+B_z^2},
\qquad
\hat{\mathbf b}=\frac{\mathbf B}{B}.
$$

$$
P_{\parallel,s}
=\hat{\mathbf b}\cdot\mathsf P_s\cdot\hat{\mathbf b},
$$

$$
P_{\perp,s}
=\frac{\operatorname{Tr}(\mathsf P_s)-P_{\parallel,s}}{2},
$$

$$
T_{\parallel,s}=\frac{P_{\parallel,s}}{n_s},
\qquad
T_{\perp,s}=\frac{P_{\perp,s}}{n_s}.
$$

The expansion used in the code is:

$$
\begin{aligned}
P_\parallel={}&P_{xx}b_x^2+P_{yy}b_y^2+P_{zz}b_z^2\\
&+2P_{xy}b_xb_y+2P_{yz}b_yb_z+2P_{zx}b_zb_x.
\end{aligned}
$$

`anisotropy_analysis.py` and `heat_flux_analysis.py` use this local projection.
Some auxiliary maps in `physical_diagnostics.py` approximate
\(P_\parallel=P_{zz}\) and \(P_\perp=(P_{xx}+P_{yy})/2\), assuming the guide
field stays mostly along \(z\).

#### 2.3. Variables

1. \(B_x,B_y,B_z\): magnetic field components.
2. \(B\): local field magnitude.
3. \(\hat{\mathbf b}\): unit vector parallel to the field.
4. \(\mathsf P_s\): thermal pressure tensor of the species.
5. \(P_{\parallel,s}\): pressure along the field direction.
6. \(P_{\perp,s}\): average of the two perpendicular pressures.
7. \(T_{\parallel,s}\), \(T_{\perp,s}\): parallel and perpendicular temperatures.

### 3. Anisotropy and parallel beta

#### 3.1. Physical rationale

The anisotropy measures which direction holds more thermal energy. Beta compares
the thermal pressure with the magnetic pressure and determines how well the
magnetic field can resist the deformation produced by the plasma.

#### 3.2. Formulas

$$
A_s=\frac{T_{\perp,s}}{T_{\parallel,s}}
    =\frac{P_{\perp,s}}{P_{\parallel,s}},
\qquad
R_s=\frac{T_{\parallel,s}}{T_{\perp,s}}=\frac{1}{A_s},
$$

$$
P_B=\frac{B^2}{2\mu_0},
\qquad
\beta_{\parallel,s}
=\frac{P_{\parallel,s}}{P_B}
=\frac{2\mu_0P_{\parallel,s}}{B^2}.
$$

Since PSC uses \(\mu_0=1\):

$$
\beta_{\parallel,s}=\frac{2P_{\parallel,s}}{B^2}.
$$

For Firehose both conventions are shown: \(A_i\) increases towards one during
relaxation, while \(R_i=1/A_i\) decreases towards one.

#### 3.3. Variables

1. \(A_s\): perpendicular/parallel anisotropy.
2. \(R_s\): inverse anisotropy.
3. \(P_B\): magnetic pressure.
4. \(\mu_0\): magnetic permeability, equal to one in code units.
5. \(\beta_{\parallel,s}\): parallel beta of the species.

### 4. Instability thresholds and the Brazil plot

#### 4.1. Physical rationale

The Brazil plot places each plasma state in the \((\beta_\parallel,A)\) plane.
Its purpose is to check whether the initial condition lies in the unstable region
and whether the evolution approaches the marginal stability threshold as a result
of particle scattering by the generated waves.

#### 4.2. Formulas used

1. Ion mirror, cold-electron bi-Maxwellian reference:

   \[
   A_i>\frac{1+\sqrt{1+4/\beta_{\parallel i}}}{2},
   \qquad
   \beta_{\perp i}(A_i-1)=\beta_{\parallel i}A_i(A_i-1)>1.
   \]

   This reference omits hot-electron effects and is not a Kappa threshold or
   the CGL mirror condition. See [Hellinger (2007), Eq. (16)](https://space.asu.cas.cz/~helinger/hell07.pdf).
   A case-specific stability claim requires the appropriate kinetic calculation.

   With the electrons included (what the mirror figures now draw, with the
   beta_e|| and A_e measured at each time), Hellinger (2007) Eq. (16) for a
   quasi-neutral proton-electron plasma reads

   \[
   \Gamma=\sum_{s=i,e}\beta_{\perp s}(A_s-1)-1
   -\frac{(A_i-A_e)^2}{2\,(1/\beta_{\parallel i}+1/\beta_{\parallel e})}>0 .
   \]

   Hot isotropic electrons raise the ion threshold slightly (1.171 -> 1.178 at
   beta_i|| = 5 for beta_e = 0 -> 8); an electron anisotropy enters the drive
   directly, e.g. A_e = 1.05 at beta_e = 8 lowers it to 1.106. It remains a
   bi-Maxwellian, marginal-stability criterion.

2. Fluid firehose:

   \[
   A_i<1-\frac{2}{\beta_{\parallel i}},
   \qquad
   \beta_{\parallel i}(1-A_i)>2.
   \]

3. Oblique firehose, kinetic approximation shown in the figure:

   \[
   A_i=1-\frac{1.4}{(\beta_{\parallel i}-0.11)^{0.55}}.
   \]

4. Ion-cyclotron, reference curve:

   \[
   A_i=1+\frac{0.43}{\beta_{\parallel i}^{0.42}}.
   \]

5. Electron whistler:

   \[
   A_e>1+\frac{0.21}{\beta_{\parallel e}^{0.6}}.
   \]

#### 4.3. Interpretation

1. Above the Mirror threshold, compressive fluctuations of \(B\) and
   mirror-type structures are expected.
2. Below the Firehose threshold, mainly transverse fluctuations and a reduction
   of the parallel pressure excess are expected.
3. Above the Whistler threshold, wave growth at electron scales and a decrease
   of \(A_e\) are expected.
4. The global trajectory uses the ratio of volume-averaged pressures, not the
   plain average of cell-by-cell ratios.

### 5. Magnetic fluctuations and Mirror structures

#### 5.1. Physical rationale

The instabilities convert free energy from the anisotropy into electromagnetic
fluctuations. Separating the parallel and perpendicular components distinguishes
a compressive response, typical of Mirror, from a transverse response, important
in Firehose and Whistler.

#### 5.2. Formulas

$$
\delta B=B-B_0,
\qquad
\frac{\delta B_{\rm rms}}{B_0}
=\frac{\sqrt{\langle(B-B_0)^2\rangle}}{B_0},
$$

$$
\frac{\delta B_{\parallel,\rm rms}}{B_0}
=\frac{\sqrt{\langle(B_z-B_0)^2\rangle}}{B_0},
$$

$$
\frac{\delta B_{\perp,\rm rms}}{B_0}
=\frac{\sqrt{\langle(B_x-\langle B_x\rangle)^2
+ (B_y-\langle B_y\rangle)^2\rangle}}{B_0}.
$$

To quantify magnetic holes:

$$
D_{\rm mirror}=1-\frac{\min(B)}{B_0},
$$

$$
f_{\rm area}
=\frac{N[B<B_0-\sigma_B]}{N_{\rm cells}},
\qquad
\sigma_B=\operatorname{std}(B).
$$

#### 5.3. Variables

1. \(B_0\): initial guide field.
2. \(\delta B\): perturbation of the field magnitude.
3. \(\langle\cdot\rangle\): spatial average.
4. \(\sigma_B\): spatial standard deviation of \(B\).
5. \(D_{\rm mirror}\): relative depth of the magnetic hole.
6. \(f_{\rm area}\): fraction of the domain occupied by low fields.

### 6. Linear growth rate

#### 6.1. Physical rationale

During the linear phase of an instability, the perturbation amplitude grows
exponentially. The slope of its logarithm gives the growth rate and allows
comparing strong, moderate, weak, Maxwellian and Kappa runs.

#### 6.2. Formulas

The amplitude is the full vector fluctuation,

$$
\delta B_{\rm rms}(t)=\langle|\mathbf B-\langle\mathbf B\rangle|^2\rangle^{1/2}
=\delta B_0 e^{\gamma t},
$$

$$
\ln\delta B_{\rm rms}(t)=\ln\delta B_0+\gamma t,
\qquad
\gamma=\frac{d}{dt}\ln\delta B_{\rm rms}.
$$

Not \(|B|-B_0\): for a transverse mode \(|B|-B_0\simeq\delta B_\perp^2/2B_0\) and
its log-slope is \(2\gamma\). The compressive (\(\delta B_\parallel\)) and
transverse (\(\delta B_\perp\)) amplitudes are fitted too.

The rms sums the PIC noise of every k with the unstable mode, so its
log-slope only reaches \(\gamma\) once the mode dwarfs the noise, i.e. near
saturation. The **reference** \(\gamma\) is therefore that of a single Fourier
mode, \(|\delta\hat{\mathbf B}(\mathbf k^*,t)|\), with \(\mathbf k^*\) the mode of
largest amplitude over the run (candidates: the strongest modes with
\(|k|d_i\le\) `K_MAX_DI_DEFAULT` at 25/50/75/100 % of the run). The rms series
are fitted on its linear phase.

The linear phase (`growth_fit.py`, shared by every script) is located on a
running median of \(\ln\delta B\): noise floor = minimum before the saturation
maximum, rise = 10-90 % of the log-rise, and inside it a contiguous interval
where the local slope stays above 80 % of its peak; of the interval around the
steepest slope and those around the steepest slope of what follows, the one
with the largest gain in e-folds wins. That skips the quiet-start build-up of
the noise floor, steeper than any instability but only ~2 \(\Omega_{ci}^{-1}\) long. Ordinary least squares
gives \(\gamma\) and its standard error; refits with other band edges and slope
fractions give the window sensitivity, and `gamma_err` combines both. On a
smoothly saturating (logistic) synthetic series the automatic fit is within
~5 % of the true rate from 15 to 2400 snapshots; for the thesis tables fix the
window with `GROWTH_T_START/END` after inspecting the fit.

Time is presented as:

$$
\tau=\Omega_{ci}t,
\qquad
\Omega_{ci}=\frac{|q_i|B_0}{m_i}.
$$

#### 6.3. Variables

1. \(\delta B_0\): initial perturbation amplitude.
2. \(\gamma\): linear growth rate.
3. \(t\): time in PSC internal units.
4. \(\tau=\Omega_{ci}t\): time normalized to the ion gyroperiod.
5. \(q_i,m_i\): ion charge and mass.

### 7. Spectral analysis

#### 7.1. Physical rationale

The spectrum identifies the wavelengths that carry the most energy and allows
checking whether the dominant mode has the scale and orientation expected for the
instability. It also separates propagation parallel and perpendicular to the
guide field.

#### 7.2. Formulas

For a two-dimensional fluctuation \(f(\mathbf x)\):

$$
\widetilde f(\mathbf k)=\mathcal F\{W(\mathbf x)f(\mathbf x)\},
\qquad
\operatorname{PSD}(\mathbf k)
=\frac{|\widetilde f(\mathbf k)|^2}{(N_1N_2)^2},
$$

$$
k_j=\frac{2\pi n_j}{N_j\Delta x_j},
\qquad
k=\sqrt{k_1^2+k_2^2}.
$$

The radial spectrum sums the power of the modes belonging to the same \(k\)
interval:

$$
E(k)=\sum_{\mathbf k\ {\rm in\ ring}\ k}
\operatorname{PSD}(\mathbf k).
$$

The power-law fit uses:

$$
E(k)=Ck^\alpha,
\qquad
\log_{10}E=\log_{10}C+\alpha\log_{10}k.
$$

For the integrated transverse magnetic spectrum:

$$
\operatorname{PSD}_{\perp}
=\operatorname{PSD}(\delta B_x)+\operatorname{PSD}(\delta B_y).
$$

#### 7.3. Variables

1. \(W\): two-dimensional Hann window used to reduce spectral leakage.
2. \(\mathbf k\): wave vector.
3. \(N_j\): number of cells along direction \(j\).
4. \(\Delta x_j\): physical cell spacing.
5. \(E(k)\): radial spectral power.
6. \(\alpha\): spectral slope.
7. \(k_\parallel,k_\perp\): components relative to the guide field, taken along
   \(z\).

#### 7.4. Mode-resolved growth rate, helicity and compressibility

Instead of a static \(E(k)\) per snapshot, \(E(k,t)\) is accumulated over the
whole run and a log-linear \(\gamma(k)\) is fitted in the growth phase of each
\(k\) ring, as in the \(\delta B(t,k)\) figure of Hellinger et al. (2018):

$$
E(k,t)\propto e^{2\gamma(k)t},
\qquad
\gamma(k)=\tfrac12\,\frac{d}{dt}\ln E(k,t).
$$

The reduced magnetic helicity and the compressibility use the two components
perpendicular to the guide field (\(\perp_1,\perp_2\)) and the parallel one:

$$
\sigma_m(k)=\frac{\operatorname{Im}\big(\widetilde B_{\perp_1}^*(k)\,
\widetilde B_{\perp_2}(k)\big)}{|\widetilde B_{\perp_1}(k)|^2+|\widetilde
B_{\perp_2}(k)|^2},
\qquad
\text{compressibility}(t)=\frac{E_\parallel(t)}{E_\parallel(t)+E_\perp(t)}.
$$

\(\gamma_\perp(k)>0\) with \(\gamma_\parallel(k)\approx0\) points to EMIC or
parallel (transverse) firehose; \(\gamma_\parallel(k)>0\) with high
compressibility points to mirror. A fit with a high \(r\)-value but negligible
final power compared to the rest of \(k\) is spectral leakage, not a physical
mode — `spectral_analysis.py` discards those bins when reporting the dominant
\(k\).

#### 7.5. What an $\omega$–$k$ diagram can actually resolve

Sampling fixes four numbers before any physics, and everything else is bounded by
them:

$$
\Delta k=\frac{2\pi}{L},\qquad
k_{\rm Ny}=\frac{\pi}{\Delta x},\qquad
\Delta\omega=\frac{2\pi}{T},\qquad
\omega_{\rm Ny}=\frac{\pi}{\Delta t_{\rm out}},
$$

with $L$ the box size, $T$ the temporal FFT window and $\Delta t_{\rm out}$ the
output cadence (not the PIC time step).

Three criteria follow:

1. **Sampling in $k$.** The number of discrete modes within the physical band of
   the instability is $\simeq(k_{\max}-k_{\min})/\Delta k$. Fitting $\omega(k)$
   needs $\gtrsim 8$; a $20\,d_i$ box gives $\Delta k\,d_i=0.31$ and therefore
   **3 modes** below $k d_i=1$.

2. **Intrinsic width.** A mode growing at $\gamma$ has a spectral width
   $\sim 2\gamma$ in $\omega$, so the branch is only readable if
   $\omega_r/\gamma\gtrsim10$. For an aperiodic mode ($\omega_r=0$: mirror,
   oblique firehose) the inequality never holds: **there is no branch to
   measure**, and the correct diagnostic is the $\gamma(k_\parallel,k_\perp)$
   map plus the $k$ spectrum, not the $\omega$–$k$ diagram.

3. **Stationarity.** The temporal FFT assumes a stationary signal. With
   $\gamma T\gtrsim3$ e-foldings inside the window, what gets transformed is the
   growth envelope and not $\omega(k)$. This is fixed by restricting the window
   to a single physical phase, or by dividing out the envelope
   ($\texttt{--degrowth per-k}$, which fits $\gamma(\mathbf k)$ mode by mode and
   divides by $e^{\gamma t}$ before the temporal transform).

`dispersion_analysis.py` evaluates all three on every run and writes
`dispersion_resolution_<plane>_<component>.json` next to the figures, with the
PASS/WARN verdict and the $L$ or $T$ that would be required.

#### 7.6. Per-instability presets

`--mode {mirror, firehose-oblique, firehose-parallel, emic, whistler, generic}`
sets the defaults each mode needs — angular band $\theta_{kB}$, physical cutoff
in $k\,d_i$, $\omega$ axis scale, $k_\perp$ reduction and temporal treatment. Any
explicit flag overrides the preset. Electron presets (whistler) are rescaled to
ion units with the mass ratio: $\omega_r/\Omega_{ci}=(\omega_r/\Omega_{ce})(m_i/m_e)$
and $k d_i=k d_e\sqrt{m_i/m_e}$.

Two practical consequences:

- Mirror and oblique firehose are searched in the band
  $\theta_{kB}\in[45°,85°]$ with `--kperp-reduction max`. Summing over $k_\perp$
  dumps the oblique peak onto the $k_\parallel$ axis, where the mode does not
  live.
- Whistlers have $\omega_r\sim0.1\text{–}0.5\,\Omega_{ce}$, i.e.
  $20\text{–}100\,\Omega_{ci}$ with $m_i/m_e=200$. A cadence designed for ion
  scales ($\Delta t_{\rm out}\sim0.07\,\Omega_{ci}^{-1}$, i.e.
  $\omega_{\rm Ny}\approx48\,\Omega_{ci}$) **aliases** them:
  $\Delta t_{\rm out}\lesssim0.013\,\Omega_{ci}^{-1}$ is required.

#### 7.7. Windows: why not in space, but yes in time

A window exists to correct the discontinuity that appears when analysing a
**non-periodic** record with a transform that assumes periodicity. The anisotropy
cases (`psc_anisotropy_case.hxx`) use `BND_FLD_PERIODIC` and `BND_PRT_PERIODIC`
on all three axes, and the dump has exactly $N$ cells for a domain $L$ (without
duplicating the boundary point). That is: **each snapshot is already an exact
period and the discrete Fourier basis is exact**. There is no leakage to correct.

Applying a window there does not remove leakage: it introduces it. Multiplying in
$x$ is convolving in $k$, and the Hann kernel is $(-\tfrac14,\tfrac12,-\tfrac14)$
in amplitude. An exact box mode gets spread like this:

| | bin $n-1$ | bin $n$ | bin $n+1$ |
|---|---|---|---|
| no window | 0 % | **100 %** | 0 % |
| Hann | 16.7 % | **66.7 %** | 16.7 % |

A third of the mode leaks into the neighbouring wavenumbers. In a $20\,d_i$ box,
where there are only 3 modes below $k d_i=1$, that amounts to smearing a third of
the useful range. Hence `--spatial-window none` is the default.

In **time** the situation is the opposite: the record starts and ends at an
arbitrary phase, does not close on itself, and without a window the sinc side
lobes sit at $-18$ dB spread across the whole $\omega$ axis — perfectly visible
on a 6-decade colour scale and easy to mistake for branches. There the window is
needed. The choice is a trade-off measured on a 772-sample record:

| temporal window | worst side lobe | main lobe width |
|---|---|---|
| rectangular | $-18$ dB | $2\,\Delta\omega$ |
| Tukey $\alpha=0.25$ | $-33$ dB | $\approx2.4\,\Delta\omega$ |
| Hann | $-48$ dB | $4\,\Delta\omega$ |

Hann doubles the effective width, and with $\Delta\omega=0.123\,\Omega_{ci}$
against $\omega_r\sim0.2$ that is exactly what cannot be spared. The default is
`--temporal-window tukey --window-alpha 0.25`, which keeps almost all the
resolution and lowers the lobes by 15 dB.

Side note: spatial detrending (`fields -= mean(axis=(2,3))`) is always correct —
it removes the $k=0$ mode, i.e. the uniform background field, not a boundary
artefact.

#### 7.8. Conventions of the ridge CSV

With `--ridge-axis k` (the default) each row is a resolved wavenumber and its
measured $\omega$, plus the full width at half maximum and the window resolution:

| column | meaning |
|---|---|
| `k_parallel_d_i` | discrete box mode, $n\,\Delta k\,d_i$ |
| `omega_over_omega_ci` | peak in $\omega$, sub-bin interpolated |
| `omega_fwhm_over_omega_ci` | full width at half maximum of the peak |
| `omega_resolution_over_omega_ci` | $\Delta\omega$ of the window |
| `resolved` | 1 only if $\omega>{\rm FWHM}$, i.e. if there is a branch |

The meaning of `resolved = 0` is literal: the peak is wider than its own central
frequency, so the row does not support a measurement of $\omega(k)$. For an
aperiodic mode every row comes out with `resolved = 0` and $\omega=0$, which is
the correct answer.

### 8. Velocity distributions and Maxwellian/Kappa fit

#### 8.1. Physical rationale

VDFs show how the particles are redistributed. A Kappa distribution has more
populated suprathermal tails than a Maxwellian; comparing both fits tells whether
the energetic particles modify growth, relaxation or transport.

#### 8.2. Formulas

$$
v_\parallel=v_z,
\qquad
v_\perp=\sqrt{v_x^2+v_y^2},
$$

$$
T_\parallel=m\,\operatorname{Var}(v_z),
\qquad
T_\perp=\frac{m}{2}
\left[\operatorname{Var}(v_x)+\operatorname{Var}(v_y)\right].
$$

Fitted one-dimensional Maxwellian form:

$$
f_M(v)=C\exp\left(-\frac{v^2}{2\sigma^2}\right).
$$

Kappa form used by the fit:

$$
f_\kappa(v)=C\left[
1+\frac{v^2}{(2\kappa-3)\sigma^2}
\right]^{-\kappa},
\qquad \kappa>1.5.
$$

The suprathermal fraction is estimated as:

$$
f_{\rm supra}
=\frac{\sum w_p\,[|\mathbf v_p|>3v_{\rm th}]}
{\sum w_p}.
$$

#### 8.3. Variables

1. \(v_x,v_y,v_z\): particle velocity components; in the non-relativistic regime
   they are approximated by the PSC normalized moments.
2. \(w_p\): statistical weight of the particle.
3. \(\sigma\): fitted width of the distribution.
4. \(\kappa\): index controlling the strength of the suprathermal tail.
5. \(C\): normalization amplitude of the fit.
6. \(v_{\rm th}\): three-dimensional thermal scale computed from the variances.

### 9. Diamagnetic current

#### 9.1. Physical rationale

A perpendicular pressure gradient produces opposite ion and electron drifts and
therefore a current. In Mirror structures this current helps to spatially sustain
the depressions and enhancements of the magnetic field.

#### 9.2. Formulas

$$
\mathbf J_{{\rm dia},s}
=\frac{\mathbf B\times\nabla P_{\perp,s}}{B^2},
$$

the sign for which \(\mathbf J\times\mathbf B=\nabla_\perp P_\perp\). In the \(YZ\)
plane, the out-of-plane component is:

$$
J_{{\rm dia},x,s}
=\frac{
B_y\,\partial P_{\perp,s}/\partial z
-B_z\,\partial P_{\perp,s}/\partial y
}{B^2},
$$

with \(P_{\perp,s}\) the thermal pressure (bulk flow removed) projected on the
local field and the gradients taken in \(d_e\), so \(J\) is in code units.

$$
J_{\rm dia,total}=J_{{\rm dia},i}+J_{{\rm dia},e}.
$$

A Gaussian filter is applied before computing gradients, to reduce PIC
statistical noise. For that reason the resulting current is a diagnostic of
coherent structure, not a measurement of cell-by-cell fluctuations.

#### 9.3. Variables

1. \(P_{\perp,s}\): perpendicular pressure of each species.
2. \(\nabla P_{\perp,s}\): spatial pressure gradient.
3. \(\mathbf J_{{\rm dia},s}\): diamagnetic current.
4. \(y,z\): coordinates of the simulation plane.

### 10. Heat flux

#### 10.1. Physical rationale

The heat flux measures thermal energy transport. It determines whether the
relaxation of the anisotropy only redistributes energy between directions or also
transports it spatially.

#### 10.2. Formulas

PSC does not deposit third moments, so the heat flux is measured from the
particles of the prt window (`heat_flux_analysis.py`, `make heatflux`). The
window is split into blocks; in each block the bulk velocity \(\mathbf U\) and
the field direction \(\hat{\mathbf b}\) are those of the block:

$$
\mathbf c_p=\mathbf v_p-\mathbf U,\qquad
\mathbf q=\frac{m}{2}\langle c_p^2\,\mathbf c_p\rangle_w,\qquad
q_\parallel=\mathbf q\cdot\hat{\mathbf b},\qquad
q_\perp=|\mathbf q-q_\parallel\hat{\mathbf b}|,
$$

normalised by the free-streaming scale

$$
q_0=\tfrac32\,T\,v_T,\qquad v_T=\sqrt{2T/m},\qquad T=(T_\parallel+2T_\perp)/3 .
$$

`v = u/gamma` (PSC stores \(u=\gamma v\)). For kappa = 3 the sixth moment of
the ideal distribution diverges, so the variance of \(q\) is infinite: the
moments are also computed over \(|\mathbf c|\le s_{\max}\sqrt{T/m}\)
(\(s_{\max}=4,6,8\)), recentred on the kept particles; quote a truncated value
with its \(s_{\max}\). The sampling error comes from disjoint subsamples. The
mean of \(|q_\parallel|/q_0\) over \(N\) particles of a VDF *without* heat flux is
not zero but the floor \(\sqrt{2/\pi}\sqrt{10}/(3\sqrt2)/\sqrt N\approx0.59/\sqrt N\)
(Maxwellian), drawn on the figure; only values clearly above it are a
measured flux.

The earlier moment-based maps \(P_\parallel U_\parallel\) were a convective
enthalpy term, not a heat flux, and have been removed.

#### 10.3. Variables

1. \(\mathbf c_p\): peculiar velocity relative to the block bulk flow.
2. \(\hat{\mathbf b}\): block-mean direction of the local magnetic field.
3. \(\langle\cdot\rangle_w\): particle-weight-weighted average.
4. \(q_0\): free-streaming heat flux scale.
5. \(s_{\max}\): truncation radius in thermal units.

### 11. Spatial correlations

#### 11.1. Physical rationale

The correlations check whether anisotropy, field, density and current belong to
the same physical structure. For example, an anticorrelation between density and
field magnitude is an expected signature of Mirror structures.

#### 11.2. Formula

For two maps \(X\) and \(Y\), the code uses the Pearson coefficient:

$$
r_{XY}
=\frac{\sum_j(X_j-\bar X)(Y_j-\bar Y)}
{\sqrt{\sum_j(X_j-\bar X)^2}
 \sqrt{\sum_j(Y_j-\bar Y)^2}}.
$$

Among others, the following are computed:

$$
r(A,|\delta B|),\quad
r(A,B),\quad
r(A,J_{\rm dia}),\quad
r(A,n),
$$

with \(A\) and \(n\) of the driven species. The structure analysis
(`structures_analysis.py`) adds \(r(\delta n_i/n_i,\delta|B|/B_0)\), the
skewness of \(|B|\) (negative: holes; positive: peaks) and the total-pressure
balance \(\mathrm{std}(B^2/2+P_{\perp i}+P_{\perp e})/\mathrm{std}(B^2/2)\).

#### 11.3. Interpretation

1. \(r=1\): perfect positive linear correlation.
2. \(r=-1\): perfect linear anticorrelation.
3. \(r\approx0\): no linear relation; does not rule out a non-linear one.

### 12. Energy balance

#### 12.1. Physical rationale

Energy tracking checks that the growth of the fields comes from the particle
energy, and helps detect numerical errors or inconsistencies between snapshots.

#### 12.2. Formulas

$$
E_{\rm bulk}
=\frac{m}{2}|\langle\mathbf v\rangle|^2,
$$

$$
E_{\rm thermal}
=\frac{m}{2}
\left\langle|\mathbf v-\langle\mathbf v\rangle|^2\right\rangle,
$$

$$
E_{\delta B}
=\frac{1}{2}\langle(B-B_0)^2\rangle,
$$

$$
E_{\rm total}
=E_{\rm bulk}+E_{\rm thermal}+E_{\delta B},
\qquad
\epsilon_E(t)=\frac{E_{\rm total}(t)-E_{\rm total}(0)}
{E_{\rm total}(0)}.
$$

This is a diagnostic balance of the available quantities, not the complete
electromagnetic energy: it does not explicitly include all of the electric field
energy nor all species in every term.

The conservation diagnostic is the global DiagEnergies budget
(`energy_conservation.py`): \(E_E+E_B+K_i+K_e\) without detrending, with the
criterion that matters for the physics,
\(\max|\Delta E_{\rm tot}|/\max|\Delta E_{\rm exchanged}|\) (the error relative
to the energy the instability moves). `energy_exchange.py` then tells which
species gives or takes it and through which channel:

$$
\frac{dK_s}{dt}=\int\mathbf J_s\cdot\mathbf E\,dV,\qquad
\mathbf J_s\cdot\mathbf E=J_{\parallel s}E_\parallel+\mathbf J_{\perp s}\cdot\mathbf E_\perp ,
$$

integrated in time and compared with \(\Delta K_s/V\) from DiagEnergies
(\(V=2E_B(0)/B_0^2\)).

#### 12.3. Variables

1. \(E_{\rm bulk}\): kinetic energy of the mean flow.
2. \(E_{\rm thermal}\): thermal kinetic energy.
3. \(E_{\delta B}\): magnetic fluctuation energy.
4. \(E_{\rm total}\): diagnostic sum.
5. \(\epsilon_E\): relative variation with respect to the first snapshot.

### 13. Moment validation

#### 13.1. Physical rationale

Before interpreting an instability, it is verified that the distribution was
actually initialized with the requested density, drift, temperature and
anisotropy. This test separates an initialization problem from a later physical
effect.

#### 13.2. Formulas

$$
n_{\rm measured}
=\frac{N_p\,C_{\rm ori}}{N_{\rm cells}},
$$

$$
\langle v_j\rangle_w
=\frac{\sum_p w_p v_{j,p}}{\sum_p w_p},
$$

$$
T_j=m\,\operatorname{Var}_w(v_j),
\qquad
v_{{\rm th},j}=\sqrt{\frac{T_j}{m}},
$$

(`physical_diagnostics.py` uses the central mixed moment
\(T_j=m(\langle u_jv_j\rangle-\langle u_j\rangle\langle v_j\rangle)\), PSC's
moment convention, which reduces to the variance for \(|u|\ll c\)),

$$
\operatorname{relative\ error}
=100\frac{|X_{\rm measured}-X_{\rm expected}|}{|X_{\rm expected}|}.
$$

#### 13.3. Variables

1. \(N_p\): number of macroparticles of the species.
2. \(C_{\rm ori}\): `cori` weight factor used by PSC.
3. \(N_{\rm cells}\): total number of cells.
4. \(w_p\): weight of each macroparticle.
5. \(X\): any validated quantity.

### 14. Mapping between scripts and diagnostics

1. `anisotropy_analysis.py`: sections 1 to 4; computes thermal pressure,
   projection onto the local field, \(A_s\), \(\beta_{\parallel,s}\), thresholds
   and Brazil plots.
2. `fluctuationofmagneticfiel.py`: section 5; generates normalized magnetic
   fluctuation maps.
3. `structures_analysis.py`: sections 5 and 11; catalogue of magnetic holes
   and peaks, |B| skewness, density correlation, pressure balance, lifetime
   (replaces `mirror_physics.py`, now in `legacy/`).
4. `spectral_analysis.py`: section 7; computes FFT, PSD, radial spectrum and
   slope (reused by `physical_diagnostics.py`), plus the mode-resolved
   `gamma(k)`, the helicity \(\sigma_m(k)\) and the compressibility described in
   7.4.
5. `plot_prt.py`: section 8; builds 2D VDFs, distribution evolution and the
   Maxwellian/Kappa comparison. The qualitative 3D visualizations live in
   `legacy/` and are not part of the maintained workflow.
6. `diamagnetic_current.py`: section 9; computes \(J_{{\rm dia},i}\),
   \(J_{{\rm dia},e}\) and the total current.
7. `heat_flux_analysis.py`: section 10; third central moment of the particle
   VDF in the local frame, truncated, with sampling error and noise floor.
8. `physical_diagnostics.py`: integrates sections 3 to 12, creating tables,
   maps, correlations, fits, growth rate and energy balance.
9. `validate_moments.py`: section 13; verifies initial density, drift,
   temperature and anisotropy using particle files.
10. `compare_physical_cases.py`: compares the same quantities across runs; it is
    only physically valid if the same anisotropy definition, driving species and
    time normalization are kept.
11. `data_reader.py`: applies no physical formula; centralizes the reading,
    assembly and selection of HDF5 datasets.
12. `psc_units.py`: defines masses, guide field, frequencies, spatial scales,
    initial temperatures and the conversion from steps to \(\Omega_{ci}t\).
13. `linear_theory.py`: solves the linear kinetic dispersion relation for
    parallel modes (bi-Maxwellian and bi-kappa) and produces the CSV consumed by
    `polarization_dispersion.py --theory-csv`. Without it, `gamma_theory` and
    `relative_difference_pct` come out NaN and there is no PIC↔theory validation.
14. `vdf_spatial.py`: spatially resolved VDF inside the prt window, using the
    positions that are present in the prt files.
15. `prt_region_field_cut.py`: locates the prt window on the field fluctuation
    maps and draws a 1D cut across it.
16. `prt_region_bfield_stats.py`: time series and time average of \(\langle|B|\rangle\),
    \(B_{\rm rms}\) and \(\delta B_{\rm rms}\) (par/perp) inside the prt window,
    with the PIC noise floor removed (see below).
17. `growth_fit.py`: section 6; the single linear-phase fit (gamma, gamma_err).
18. `energy_conservation.py`: section 12; global DiagEnergies budget.
19. `energy_exchange.py`: section 12; \(J_s\cdot E\) per species and channel.
20. `field_residuals.py`: div B of the Yee snapshots and PSC's Gauss /
    continuity checks from the job log.
21. `estimator_consistency.py`: the four anisotropy estimators (particles in
    the B0 and local frames, moments in the window and in the domain) on one
    time axis, with the fraction of cells the Brazil-plot filter rejects.
22. `convergence_study.py`: observables of several runs vs dx, dt, ppc, box
    and seed; realization mean and spread.
23. `synthetic_run.py`: a PSC-format run with known answers, to validate the
    chain end to end.
24. `plasma_physics.py`: shared formulas (pressure projection, thermal
    pressure, diamagnetic current, reference thresholds, u/v kinematics).

## Linear theory and spatial VDF

Before comparing PIC with theory, the theoretical curve must be generated:

```bash
make theory-self-test
```

```bash
make theory CASE=firehose_bikappa3_bigbox40
```

```bash
make polarization DATA_DIR=/path CASE=firehose_bikappa3_bigbox40 THEORY_CSV=../analysis_results/firehose_bikappa3_bigbox40/04_spectra/linear_theory.csv
```

`make theory-self-test` checks the solver against three limits with known answers
(the \(Z'\) identity, the convergence \(Z_\kappa \to Z\) as \(O(1/\kappa)\), and
the analytical parallel-firehose threshold \(\beta_\parallel - \beta_\perp = 2\)).
The solver covers **parallel propagation only**: the mirror mode is oblique and
aperiodic and does not come out of this relation.
These self-tests do not validate finite-Kappa temperatures or root convergence.
The audit records unresolved thermal-scale and polarization-CSV issues; generated
theory curves must not yet be treated as validated quantitative PIC comparisons.

For the spatially resolved VDF and the location of the prt window:

```bash
make prt-region DATA_DIR=/path CASE=mirror_bikappa3_moderate
```

```bash
make vdf-spatial DATA_DIR=/path CASE=mirror_bikappa3_moderate
```

Magnetic-field statistics inside the same window, time-averaged:

```bash
make prt-bfield DATA_DIR=/path CASE=mirror_bikappa3_moderate \
    PRT_BFIELD_FLAGS='--avg-window 100 158'
```

`prt_region_bfield_stats.py` removes the PIC noise in two steps: a spectral
low-pass on the full periodic domain (\(|k| \le k_c\), default half the grid
Nyquist), then a quadrature subtraction of the noise floor. With
`--floor tracked` (default) the floor follows the power above \(k_c\), which
carries no physical signal, scaled by the low/high-band ratio measured in a quiet
window before growth; this follows the rise of the floor caused by numerical
heating. The quiet window (`--noise-window`, in \(\Omega_{ci}t\)) must end before
linear growth: for the whistler cases that is a few snapshots, so set it by hand
after looking at the figure. Outputs go to `09_physical_diagnostics/`.

`vdf_spatial.py` splits the particles into `hole` / `ambient` / `peak` according
to the \(|B|\) of their cell and compares \(A\) between populations **against the
sampling noise** (\(\sigma_A/A \simeq \sqrt{3/N}\)): it reports the difference in
units of \(\sigma\), so that an apparent separation in a colour map is not
mistaken for a measurement. The anisotropy is taken relative to the local field
\(\hat{b}\), not to the global \(z\).

### 12. Spatially resolved kappa index: estimator, b-binned profiles, closures

#### 12.1. Physical rationale

The adiabatic Liouville mapping of a bi-kappa along a mirror structure
conserves \(\kappa\) on the passing branch — only \(\theta_\perp\) is
renormalised, \(\theta_{\perp,\rm eff}^2 = \theta_\perp^2\, b /
[1 - A_0(1-b)]\), with \(b = B/B_{\rm ref}\) and maximum depth
\(a_{\max} = 1 - 1/A_0\). Any measured variation of \(\kappa_{\rm eff}(b)\)
is therefore a signature of how the trapped domain
(\(\sin^2\alpha > b\)) is filled, or of non-adiabatic dynamics.

#### 12.2. The estimator (`kappa_eff.py`)

\(\kappa_{\rm eff}\) comes from the moment ratio
\(K = \langle s^4\rangle/\langle s^2\rangle^2\) with the velocities
**whitened** per component (\(s_j = v_j/\sigma_j\), drift subtracted) and
**truncated** at \(s \le s_{\max}\) (default 6):

* whitening removes anisotropy aliasing — a bi-Maxwellian with \(A \ne 1\)
  would otherwise report a spurious finite kappa;
* truncation handles the divergence of \(\langle v^4\rangle\) for
  \(\kappa \le 5/2\) (the \(\kappa = 3\) runs!) and mimics an instrument's
  finite energy range. The truncated relation \(K_t(\kappa, s_{\max})\) is
  inverted numerically; the untruncated limit is
  \(\kappa = \tfrac52 (K-1)/(K-\tfrac53)\).

The same estimator is applied to particles
(`kappa_eff_from_velocities`) and to theoretical distributions on a
\((v_\parallel, v_\perp)\) quadrature grid (`kappa_eff_from_grid`), so
theory, simulation and (eventually) instrument data share one definition.
Validation against loader-consistent synthetic bi-kappas lives in
`test_kappa_eff.py`; `make kappa-eff-self-test` runs a quick check.

#### 12.3. b-binned profiles (`vdf_spatial.py`, paper fig. 7)

Each particle is tagged with \(b = |B|_{\rm local}/B_{\rm ref}\)
(\(B_{\rm ref}\) = high percentile of the window \(|B|\), per snapshot) and
classified trapped/passing with \(\sin^2\alpha > b\) in the drift-subtracted
local-\(\hat b\) frame. Binning in \(b\) yields \(n(b)\), \(T_\perp/T_\parallel(b)\),
the trapped fraction (with the isotropic reference \(\sqrt{1-b}\)) and
\(\kappa_{\rm eff}(b)\), per snapshot and aggregated (superposed epoch in
field space; kappa aggregates in \(1/\kappa\), where the Maxwellian limit is
exactly 0). Outputs: `vdf_b_profile_step*.csv`, `vdf_b_profile_aggregate.csv`
and `vdf_b_profiles.png`. Tune with `VDF_FLAGS='--b-bins 12 --b-ref-percentile 98
--s-max 6 --kappa-boot 24'`. When aggregating, restrict `--steps` to the
saturated phase — mixing the linear stage in smears the profiles.

#### 12.4. Liouville closures (`liouville_kappa.py`, paper fig. 2)

Computes the closed-form mapped passing branch plus the three trapped-domain
closures — case 1 `empty` (\(f=0\)), case 2 `flat` (continuity, flat along
\(v_\parallel\)), case 3 `own` (own bi-kappa \(\kappa_t\); with
\(\kappa_t=\kappa_0\) it is the seamless filling and
\(\kappa_{\rm eff}(b) = \kappa_0\) exactly) — and their \(n\), \(A\),
trapped-fraction and \(\kappa_{\rm eff}\) profiles vs \(b\), overlayable on
the measured fig. 7:

```bash
make theory-liouville CASE=mirror_bikappa3_moderate
make liouville-self-test
```

The self-test verifies the closed form against the raw Liouville mapping
point-wise (the kappa-invariance theorem), the depth bound \(b > 1 - 1/A_0\),
and that the closure signatures in \(\kappa_{\rm eff}\) behave as documented.

## Reconnection runs: field evolution and the B–kappa correlation

The double-Harris reconnection runs (`psc_reconnection`,
`psc_reconnection_comparable`) are analysed by `reconnection_analysis.py`,
which replaces the legacy script of the same name. Their boxes are
rectangular and their parameters are not part of the anisotropy case matrix,
so the script carries its own two profiles and reads the grid, the box and
the output cadence back from the snapshots themselves; what was detected is
recorded in `reconnection_summary.json`.

```bash
make reconnection DATA_DIR=/path/to/run RECONNECTION_PROFILE=reconnection_comparable
```

writes into `10_reconnection/`:

- `reconnection_field_timeseries.csv` — per field snapshot,
  \(\langle|B|\rangle/B_0\), \(\min|B|/B_0\) and \(\delta B_{\rm rms}/B_0\)
  over two regions: the **prt output window** (the volume the particle
  output actually samples) and a band of `±SHEET_HALF_WIDTH` (default
  \(2\,d_i\)) around the **perturbed sheet** at \(y=+L_y/4\); plus
  \(\max_z|B_y|/B_0\) on the sheet as a reconnected-flux / tearing proxy.
- `reconnection_kappa_timeseries.csv` — per particle snapshot, the
  effective kappa of the selected species from the truncated, whitened
  estimator of `kappa_eff.py`, in the field-aligned frame of the
  window-mean field of the matching field snapshot. Reported also as
  \(1/\kappa\) (Maxwellian limit = 0, as in `kappa_evolution.py`).
- `b_kappa_correlation.json`, `b_kappa_evolution.png`,
  `b_kappa_scatter.png` — Pearson (on \(1/\kappa\)) and Spearman
  correlations of \(\langle|B|\rangle(t)\) against \(\kappa_{\rm eff}(t)\)
  at the particle cadence, for both regions, with a small lag scan when
  enough snapshots exist.
- `reconnection_overview.png`, `reconnected_flux.png` — \(B_z/B_0\) and
  \(|B|/B_0\) maps with in-plane field lines at representative times, and
  the flux-proxy evolution.

Three caveats the outputs repeat on purpose: the prt window of
`psc_reconnection_comparable` covers the **inflow region between the
sheets**, not the X-point, so \(\kappa_{\rm eff}(t)\) characterises the
plasma feeding the reconnection (moving the window is a simulation-case
decision, not an analysis option); the correlation p-values assume
independent samples while consecutive snapshots are autocorrelated; and a
correlation between two series driven by the same instability clock is not
causation. Pass options through `RECONNECTION_FLAGS`, e.g.
`RECONNECTION_FLAGS='--species electron --kappa-boot 24 --dt-code 0.33'`.
Regression tests: `python -m unittest test_reconnection_analysis`.

## Technical documentation

For the internal file structure, HDF5 datasets and the responsibilities of each
script, see:

```text
CodeforAnalisys/ANALISIS_ESTRUCTURA.md
```

## Revision 6: evidence and estimator contracts (2026-09-28)

Use a **new results directory**. Revision 6 changes VDF display coordinates,
structure definitions, growth uncertainty and metadata. Preserve v5 as the
historical baseline. Existing local modal-growth fixes are retained.

The ordered runner isolates each stage, promotes its products only after the
command succeeds, and writes atomic completion records with source/input/output
fingerprints, configuration, logs, elapsed time and peak memory. It runs the
initial-state preflight before anything else and passes an accepted modal phase
to the spectral stages (recorded as `window_source` in `pipeline.json`). A stage
may never overwrite a product of another stage: `physics` already writes the
DiagEnergies budget, so `energy-if-present` is only accepted for a run analysed
without `physics` (the two used to overwrite each other, and every `--resume`
then re-ran both). A completion marker describes execution success; the report
separately assesses scientific evidence. No run is certified from an exit code
alone.

From the repository root:

```bash
.venv/bin/python CodeforAnalisys/run_pipeline.py \
  --data-dir /path/to/raw/run --case mirror_bimaxwellian_moderate \
  --results-root "$PWD/analysis_results/v6"
```

Add `--resume` only for the same source, inputs and configuration; a re-run
stage also re-runs the stages ordered after it (`spectral` after `physics`). To
select a subset, use e.g. `--stages manifest physics spectral`; manifest remains
a required preflight. Optional stages: `theory` (parallel linear theory for the
polarization overlay; refused for mirror cases), `theory-liouville` and
`energy-if-present`.

`--jobs N` runs up to N independent stages at once (longest first; `spectral`
waits for `physics`). `--launcher 'srun --nodes=1 --ntasks=1 --exclusive'`
puts each stage on its own node while the bookkeeping stays in one process;
`cosma_jobs/analisis/reanalysis_v6_all.sh` uses exactly this. `--keep-going`
still runs the stages independent of a failed one; the report is always
written, and the exit code is non-zero if anything failed or did not run. `--make-option LOGS=/path/to/run.out` passes the actual runtime log.
`PSC_ANALYSIS_CONFIG=/path/to/runtime.json` supplies verified run values through
the existing geometry resolver; values are not inferred from the desired result.
The runner caps numerical-library threads at one per process to prevent nested
oversubscription; benchmark worker counts on production data separately.

Audit an existing delivery without reprocessing its snapshots:

```bash
.venv/bin/python CodeforAnalisys/quality_report.py \
  results/v5/mirror_bimaxwellian_moderate \
  results/v5/mirror_bikappa5_moderate \
  results/v5/mirror_bikappa3_moderate \
  --outdir analysis_results/v6_audit
```

The report contains HTML figure links and a CSV/JSON evidence matrix. The optional
`--energy-tolerance` must come from an independently justified numerical error
budget. Without it, a measured energy drift is **UNVERIFIED**, not automatically
accepted, unless the energy audit below finds it FAILS on its own terms. Missing
logs, initial data, mode evidence and independent convergence remain explicit. No
tolerance is selected to make a result pass. A growth rate is PASS only for an
accepted **modal** fit: the legacy `total` series (domain rms of dB, biased low)
stays UNVERIFIED even when its fit was accepted.

**Energy audit** (`energy_audit.py`, also run by the report into
`<outdir>/energy_audit/`). From existing products only, per run: (1) the
DiagEnergies columns at t = 0 must reproduce E_s/E_B = beta_s,par (1/2 + A_s),
which checks species order, the 1/2 field factor and the volume; (2) a change of
the total energy, which a periodic box without sources cannot have, is compared
with the energy released by the driving species and FAILS at half of it; (3)
without DiagEnergies the prt-window moments FAIL when the electrons gain more
than twice what ions and magnetic fluctuations release there; (4) dx/lambda_De(t),
beta_e(t) and T_e/T_e0 at the end of the fitted linear phase. Across runs that
differ only in the distribution it reports whether the electron heating is
common-mode. It assigns no mechanism.

```bash
.venv/bin/python CodeforAnalisys/energy_audit.py results/v5/mirror_* \
  --outdir analysis_results/v6_audit/energy_audit
```

**Isotropic controls.** `energy_audit.py` pairs every anisotropic run with an
isotropic control among the given runs (same numerics and electrons, A_i = 1,
preferably the same distribution and ion thermal energy; `--control RUN=CTRL`
forces a pair) and subtracts the control's energy changes on a common time
axis: the electron heating without the numerical baseline and the
baseline-corrected closure (PASS <= 0.1, FAIL >= 0.5 of the ion release). The
subtraction assumes the numerical heating is not changed by the instability;
the raw budget keeps its own status. The controls are the cases
`psc_mirror_*_isotropic` (`src/SIMULACIONES_ANISOTROPIA.md`).

```bash
.venv/bin/python CodeforAnalisys/energy_audit.py analysis_results/v6/mirror_* \
  --outdir analysis_results/v6/quality_report/energy_audit
```

**Figure conventions** (all through `plot_style`): paper theme, 300 dpi PNG plus
PDF; spatial maps with z (along B0) horizontal, y vertical, equal aspect and
integer d_i ticks (`plot_style.spatial_axes`); k_parallel horizontal in every
k-space map; time as t Omega_ci; velocities as v/v_A after converting PSC's
u = gamma v; log axes labelled at 1-2-5 values (`plain_log_axis`); a power of
ten goes into the axis or colour-bar label instead of a floating offset; one
diverging (RdBu_r) and one sequential (viridis) colour map; Okabe-Ito series
colours.

**Figure content check.** `plot_style.save` records, next to every figure, what
each panel actually draws (`figure_qa_<script>.jsonl`): series that draw nothing
(e.g. samples isolated between NaNs with no marker — the v5 comparison of A(t)),
maps with no finite value, empty panels, and layout defects: overlapping text
(titles, labels, ticks, annotations, colour bars, legends) and legends or
annotations covering data. The report's `figures` check lists them (WARN). A
successful `savefig` is not evidence that anything was plotted or legible.

Changes to measurement conventions:

- VDF displays use converted **v**; the analytic initialization-distribution fit
  deliberately remains in **u = gamma v**, with corrected labels and CSV fields.
  A model in u is not silently reused as a model in v. Gyrotropic 2D densities use
  exact annular bin volumes and record probability excluded by the display range.
- SciPy and fallback fits use the same linear-density least-squares objective.
  Solver/fallback reasons, effective counts, conditional parameter error and
  held-out particle scores are exported. Predictive improvement does not identify
  a physical kappa population or exclude a mixture of local populations.
- Mode discovery includes early logarithmically spaced snapshots. Modal tables
  retain the strongest-power reference and separately flag the fastest accepted
  candidate; this is not automatic branch identification. Growth errors include
  an explicitly recorded HAC estimate for correlated residuals and window spread.
- Structure morphology is relative to the instantaneous mean magnitude; the
  change relative to B0 is retained separately. Thermal pressure uses co-located
  B, and magnetic pressure is smoothed after squaring. Threshold/smoothing
  sensitivity, overlap tracks, split/merge events and censored durations are
  exported. Fast advection can break a pixel-overlap track.
- Heat flux exports both a particle-weighted mean of local normalized values
  and a ratio of integrated moments. Its matched symmetric-null floor is an
  asymptotic influence-function estimate on the truncated distribution, not an
  exact detection threshold for arbitrary kappa tails. The old Maxwellian
  reference remains in the table for comparison.
- Energy closure reports common time coverage and the entire cumulative residual.
  `energy_exchange.py --time-tolerance-code VALUE` permits only explicit bounded
  endpoint snapping for verified output rounding (default zero). It does not
  repair energy drift, extrapolate distant data or validate temporal staggering.
- Comparisons reject mixed estimators/algorithm versions and avoid connecting
  large gaps or drawing an invalid growth rate as a zero-valued measurement.
- The gamma(k_par, k_perp) display zoom (`--display-kperp-max`, 0.9 d_i^-1 by
  default for ion-scale cases) is widened whenever an accepted growing cell lies
  outside it, with a note on the figure; it previously hid the only growing mode
  of the synthetic run. The axes follow the computed bins, not the angle guides.
- Initial-state validation allows max(fixed tolerance, 4 sampling standard
  errors), the latter from the measured fourth moment (kappa tails make it far
  larger than the Gaussian sqrt(2/N)). Production windows (~10^7 particles) are
  unaffected; a correctly initialised small sample no longer FAILS.
- The PSC log counts as covering the run when its last check lies within one
  check cadence of nmax (checks run every `continuity_every` steps).
- Particle VDFs in `plot_prt.py` and `vdf_spatial.py` convert u = gamma v to v
  (the labels said v while u was binned), are weighted and normalised by the
  whole population rather than the plotted range. The |v_perp| distribution is
  compared with the 2-D Maxwellian (Rayleigh) of the t = 0 variance; it used to
  be compared with a 1-D Gaussian, the wrong reference.
- Spectral indices are fitted only over a decaying range above twice the peak
  and 10x above the noise floor (>= 5 points, >= 0.3 decades, R^2 >= 0.9);
  otherwise the figure says so and `power_law_slope` is empty. The old fixed
  middle-half fit reported noise slopes such as k^+2.8.
- A kappa fit at its upper bound (80) is Maxwellian-consistent, not a
  measurement: the time series shows 1/kappa with such fits marked.
- `mirror_area_fraction` was the fraction of cells below B0 - std(|B|), ~16-25 %
  for any amplitude; it is renamed `fraction_below_B0_minus_std` and no longer
  plotted as a hole area (hole populations: `07_structures`).
- The magnetic-energy panel of `plot_prt.py` plotted (dB/B0)^2/2 labelled in
  units of B0^2/2, a factor 2 low; it now plots (dB/B0)^2 = E_dB/(B0^2/2).
- `vdf_spatial.py` adds a spatial-mixture control of the window kappa tail:
  a kappa is exactly a Gamma mixture of Maxwellian temperatures, so the
  macro-cell temperature spread alone gives kappa_mix = 5/2 + 1/CV^2(tau) for the
  whitened estimator. `mixture_fraction_of_tail` and `mixture_verdict`
  (`mixture_explains_tail` >= 0.75, `intrinsic_at_macrocell_scale` <= 0.25,
  `partial_mixture`, `no_significant_tail`) are written on the `all` rows.

See `RESULTS_V5_IMPROVEMENT_PLAN.md` section 9 for implementation status and
remaining production/research validation. Standalone diagnostic scripts remain
available; the ordered runner is the recommended reproducible workflow.
