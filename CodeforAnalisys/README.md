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
06_heat_flux/      heat flux and spatial regions
07_mirror_structures/ local |B| depressions for mirror
08_validation/     pointwise validation against particles
09_physical_diagnostics/ integrated diagnostics with the standard outputs
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
`spatial_correlations.csv`, `energy_table.csv`, plus `T_parallel/T_perp/A_i`,
`deltaB`, `mirror_holes` and `J_dia` maps, 2D VDFs, Maxwellian/Kappa fits,
growth rate, correlations and energy.

To compare already-analysed cases, for example Maxwellian vs Kappa:

```bash
make compare-physics COMPARE_CASES="maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics"
```

This produces `comparison_kappa_vs_maxwellian.csv`, `comparison_anisotropy.png`,
`comparison_deltaB.png`, `comparison_growth_rate.png`, `comparison_energy.png`
and `comparison_heat_flux.png`.

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

$$
\delta B_{\rm rms}(t)=\delta B_0 e^{\gamma t},
$$

$$
\ln\delta B_{\rm rms}(t)=\ln\delta B_0+\gamma t,
\qquad
\gamma=\frac{d}{dt}\ln\delta B_{\rm rms}.
$$

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
=\frac{\nabla P_{\perp,s}\times\mathbf B}{B^2}.
$$

In the \(YZ\) plane, the dominant out-of-plane component is:

$$
J_{{\rm dia},x,s}
=\frac{
(\partial P_{\perp,s}/\partial y)B_z
-(\partial P_{\perp,s}/\partial z)B_y
}{B^2},
$$

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

The diagnostic based directly on particles uses the third central moment:

$$
\mathbf c_p=\mathbf v_p-\langle\mathbf v\rangle,
\qquad
c_p^2=\mathbf c_p\cdot\mathbf c_p,
$$

$$
q_{\parallel}^{(p)}
=\frac{m}{2}\langle c_p^2c_{\parallel,p}\rangle_w,
\qquad
q_{\perp}^{(p)}
=\frac{m}{2}\langle c_p^2c_{\perp,p}\rangle_w.
$$

In the code, \(c_{\perp,p}=\sqrt{c_{x,p}^2+c_{y,p}^2}\). Therefore
\(q_{\perp}^{(p)}\) measures a positive perpendicular magnitude and not a signed
vector component. The full vector definition would be
\(\mathbf q=(m/2)\langle c^2\mathbf c\rangle\).

The maps built from fluid moments are transport proxies:

$$
v_\parallel=\mathbf u\cdot\hat{\mathbf b},
\qquad
\mathbf v_\perp=\mathbf u-v_\parallel\hat{\mathbf b},
$$

$$
q_\parallel^{({\rm proxy})}=P_\parallel v_\parallel,
\qquad
q_\perp^{({\rm proxy})}=P_\perp|\mathbf v_\perp|.
$$

The moment maps do not contain the full third moment and therefore must not be
interpreted as the exact kinetic heat flux. The particle calculation is the
physically more direct diagnostic.

#### 10.3. Variables

1. \(\mathbf c_p\): peculiar velocity relative to the mean flow.
2. \(c_{\parallel,p}\), \(c_{\perp,p}\): parallel and perpendicular peculiar
   components.
3. \(\langle\cdot\rangle_w\): particle-weight-weighted average.
4. \(\mathbf u\): macroscopic velocity.
5. \(q_\parallel,q_\perp\): parallel and perpendicular thermal energy transport.

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
r(A,\delta B),\quad
r(A,B),\quad
r(A,J_{\rm dia}),\quad
r(A,\rho_i).
$$

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
3. `mirror_physics.py`: sections 5 and 9; visualizes Mirror magnetic structures
   and the associated current.
4. `spectral_analysis.py`: section 7; computes FFT, PSD, radial spectrum and
   slope (reused by `physical_diagnostics.py`), plus the mode-resolved
   `gamma(k)`, the helicity \(\sigma_m(k)\) and the compressibility described in
   7.4.
5. `plot_prt.py`: section 8; builds 2D VDFs, distribution evolution and the
   Maxwellian/Kappa comparison. The qualitative 3D visualizations live in
   `legacy/` and are not part of the maintained workflow.
6. `diamagnetic_current.py`: section 9; computes \(J_{{\rm dia},i}\),
   \(J_{{\rm dia},e}\) and the total current.
7. `heat_flux_analysis.py`: section 10; computes the spatial proxies
   \(P_\parallel v_\parallel\) and \(P_\perp v_\perp\).
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

## Technical documentation

For the internal file structure, HDF5 datasets and the responsibilities of each
script, see:

```text
CodeforAnalisys/ANALISIS_ESTRUCTURA.md
```
