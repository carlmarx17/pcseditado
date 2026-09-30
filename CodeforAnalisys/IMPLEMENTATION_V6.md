# Analysis v6 implementation record

Date: 2026-09-28. Historical products in `results/v5` and simulation parameters
were preserved. Pre-existing local modal-growth and comparison edits were
extended, not discarded.

## Delivered

- Shared version/provenance, strict JSON, sampling, geometry and density utilities.
- Source/input fingerprints, reproducibility metadata and explicit scientific
  evidence statuses; HTML/CSV/JSON audit for the three delivered mirror cases.
- Correct velocity-space VDF displays and explicitly labelled u-space fits;
  cylindrical normalization, comparable solver objectives and held-out scores.
- Earlier modal candidate discovery, separate fastest-growth candidate,
  correlated-residual uncertainty and estimator-version comparison guards.
- Common-time energy closure with explicit bounded rounding tolerance and full
  residual coverage; the energy anomaly is not detrended or concealed.
- Mean-centred structures, co-located pressure balance, sensitivity sweeps and
  conservative split/merge tracking.
- Explicit heat-flux averaging, effective counts and a matched asymptotic null
  noise estimate, with its limitations recorded.
- Ordered isolated stage execution, completion records, resumability, exclusive
  output locking and timing logs. See `README.md` for the commands.

## Verification

- Known-answer and invariance tests cover annular normalization, Fourier
  translation/scaling/Parseval/Nyquist, early-mode discovery, weighted and rotated
  heat-flux moments, correlated errors, missing evidence, mixed estimators,
  held-out scores and structure splits.
- The synthetic diagnostic suite exercises the actual scripts as subprocesses.
- An additional runner smoke test completed manifest, energy, residuals,
  structures and energy exchange. A separate manifest resume check skipped the
  completed stage successfully using matching source/input/output fingerprints.
- The synthetic structure map was visually inspected. `git diff --check` passed.
- The knowledge graph was rebuilt and enriched; graphify reported existing
  extraction warnings and a report-refresh KeyError. The enriched graph JSON was
  saved, but that report refresh was not successful.

## Limits and outstanding work

This is a substantial processing revision, not completion of every research item
in the plan. Original COSMA data/logs are required to regenerate production
figures and resolve the energy anomaly. New convergence runs, oblique kinetic
solver integration, rigorous mixture/model-selection validation, full nonlinear
uncertainty calibration, production I/O/cache benchmarking and final publication
QA remain outstanding. The detailed item-by-item status is in section 9 of
`RESULTS_V5_IMPROVEMENT_PLAN.md`.

The local v5 evidence report deliberately leaves all three runs UNVERIFIED:
passing synthetic tests does not establish physical validity of those runs.

Final full-suite result on the completed code: **153 tests passed, 14 subtests
passed in 60.83 seconds**, using `.venv/bin/python -m pytest -q CodeforAnalisys`.

## Follow-up (2026-09-29): production path, energy audit and QA

### Defects found and corrected

- **Production regeneration was blocked.** `reanalysis_v5_all.sh` refuses any
  checkout whose conventions version is not 5, and there was no v6 job. Added
  `cosma_jobs/analisis/reanalysis_v6_all.sh` (see `cosma_jobs/README.md`, E).
  It was run end to end locally on synthetic runs with the production folder
  names through a stand-in `srun`, under bash 3.2; that run caught an
  empty-array expansion that aborts `set -u` on bash < 4.4.
- **Runner.** Only 5 of 15 stages had been exercised. The full run showed
  `energy-if-present` and `physics` writing the same four files, so every
  `--resume` re-ran both (including the most expensive stage). A stage can now
  never overwrite another stage's products; the pair is rejected up front.
  Added `--jobs`, `--launcher` (srun per stage), `--keep-going`, per-process
  peak memory (was cumulative over children), early case validation, `theory`
  refused for mirror cases, a lock warning on filesystems without `flock`, and
  re-running of stages ordered after a re-run stage.
- **gamma(k) map hid the growing mode.** The default 0.9 d_i^-1 k_perp display
  zoom cropped the only accepted cell of the synthetic run (k_perp d_i = 0.94),
  leaving an empty map although the CSV held gamma = 0.236 (input 0.25). The
  zoom now widens to every accepted cell and says so; axes follow the bins.
- **Evidence report.** An accepted fit of the legacy `total` series (domain rms,
  biased low) counted as PASS; the energy check ignored the prt-window moments
  when `diag.asc` was missing; the PASS reason of the initial state read
  "validation required". All corrected.
- **Initial validation** used fixed 2 % tolerances whatever the sample size, so
  the known-answer synthetic run FAILED; tolerances are now max(fixed, 4
  sampling standard errors from the measured fourth moment). **Log coverage**
  required the last check at exactly nmax instead of within one check cadence.
- Figure-level: an empty `vdf_spatial` macro-cell panel now says why; a legend
  proxy in `growth_rate_vs_k` used a marker the data do not use; Spanish verdict
  strings in `electron_energy_trend.csv` are now English.
- The 16 stale grid/ppc header comments of the case files now state the values
  actually run (576², 1000 ppc; job scripts confirm, manifests detect 576²). No
  define changed. The parity checker is down from 18 to 2 findings.

### Added

- `energy_audit.py`, used by the report: diagnostic mapping, closure against
  the driver's release, window proxy, dx/lambda_De(t), beta_e(t), early-time
  T_e error, cross-run common-mode test.
- Figure content records at every `plot_style.save` and a `figures` check.
- Spatial-mixture control of the kappa tail in `vdf_spatial.py`
  (`vdf_validation.mixture_tail_budget`), validated on an exact Gamma mixture of
  Maxwellians (kappa_mix 5.12 vs kappa_eff 5.09), a true kappa and a Maxwellian.
- `test_evidence_tooling.py`: 17 known-answer tests (audit, figure check,
  runner ordering/collision/resume/keep-going through a stand-in make, shot-noise
  tolerance, log cadence, map zoom, mixture control). `pytest` joined
  `requirements.txt`, since the COSMA preflight runs the suite.

### Energy audit of the v5 deliveries (most important result)

`analysis_results/v6_audit/energy_audit/` (regenerable with the command in the
README). All three mirror-moderate runs **FAIL** the energy budget:

| Run | T_e end / T_e0 | e gain / (ion + field release), window | dx/lambda_De start → end | A_e end | Global DiagEnergies |
|---|---|---|---|---|---|
| bi-Maxwellian | 8.32 | 6.4 | 8.71 → 3.02 | 0.98 | not delivered |
| bi-kappa 5 | 8.43 | 6.5 | 8.72 → 3.00 | 0.98 | +64.6 % total, 5.5x the ion release; mapping PASS |
| bi-kappa 3 | 8.55 | 6.2 | 8.75 → 2.99 | 0.98 | not delivered |

- The DiagEnergies columns reproduce the initial partition (E_i/E_B = 12.498,
  E_e/E_B = 1.492, expected 12.5 / 1.5 with a 0.4 % relativistic correction):
  the anomaly is **not** a column, factor or volume error.
- The window and global electron heating agree within 2.3 %: it is domain-wide.
- The heating differs by only 3.1 % between the three distributions: it belongs
  to the shared numerical setup, not to the distribution being compared.
- It is isotropic, starts at t = 0 and accelerates (second half of the run heats
  2.2x more than the first). A classical finite-grid instability slows as
  lambda_De approaches dx; this one does not, so the mechanism is **not**
  assigned. Candidates remain dx/lambda_De = 8.7 with first-order shapes and
  particle noise.
- Early-time error: T_e/T_e0 = 1.23 at t Omega_ci = 10, 1.45–1.48 at 20,
  ~2.0 at 40, and 2.7–2.9 at the end of the legacy fitted linear phase
  (t Omega_ci ≈ 63–68). beta_e ends near 8.4 instead of 1. Late-time electron
  pressure, pressure balance and saturation are affected most; a linear growth
  rate is only defensible with this T_e drift stated alongside it.

Discriminating reruns (a proposal; not submitted, parameters unchanged): short
runs of `mirror_bimaxwellian_moderate` to t Omega_ci ≈ 40 (`PSC_NMAX=300000`)
with (a) the baseline, (b) `PSC_NICELL=4000`, (c) `PSC_NGRID=1152` (dx/lambda_De
4.3, dt halves by CFL). Noise heating scales with 1/ppc, grid heating with dx;
both are environment overrides of the existing executable. A second-order shape
(`PscConfig2ndSingle`) would need a new executable and a case-integrity review.

### Still outstanding

- Production regeneration: submit `reanalysis_v6_all.sh` on COSMA.
- The energy anomaly needs the reruns above; post-processing cannot repair it.
- Oblique kinetic theory for the mirror (solver choice and validation), full
  nonlinear confidence intervals, heat-flux floor calibration on production
  tails, advection-aware tracking, I/O benchmarking and the final thesis figure
  selection remain research or production work.
- Migration of the two monolithic reconnection cases (structural; needs a
  decision per the case-integrity rules).

## Follow-up 2 (2026-09-29): isotropic controls and publication review

### Isotropic controls of the numerical heating

- Cases `psc_mirror_{bimaxwellian,bikappa5,bikappa3}_isotropic`: the moderate
  twins with A_i = 1 and the same ion thermal energy (beta_i,par = 25/3), same
  numerics and electrons; CMake targets, case table
  (`src/SIMULACIONES_ANISOTROPIA.md`) and job scripts
  (`cosma_jobs/simulacion/sim_mirror_*_isotropic.sh`, `PSC_ENERGIES_EVERY=500`).
  The parity checker accepts them as distribution twins (still only the two
  reconnection findings).
- `energy_audit.py` pairs each run with its control and subtracts the
  control's energy changes: baseline-corrected electron heating and closure
  (PASS <= 0.1, FAIL >= 0.5 of the ion release; the raw status is kept). The
  evidence report adds an `energy_baseline` check and figure; the v6 job takes
  the controls through `EXTRA_RUNS` and reports over the whole v6 tree.
- Known-answer tests: exact closure recovered, wrong control rejected,
  same-distribution preference, explicit pairing.

- Analysis profiles `mirror_*_isotropic` added to `psc_units.py`. All 28
  profiles of maintained cases were checked against their case files and job
  scripts (beta, A, kappa, box, mi/me, grid, ppc, nmax, cadences) and match;
  `test_profiles.py` now enforces it. A synthetic control run goes through the
  pipeline and is paired automatically; a control that loses as much ion energy
  as its run leaves nothing to close and is reported UNVERIFIED, not FAIL.

### Publication review of the figures

- Automatic layout checks at every save: overlapping text of any kind, crowded
  ticks, legends or annotations covering data. On the synthetic run the
  pipeline went from 111 findings in 129 figures to 0 in 147 (all figures now
  pass through the check; three scripts used to bypass the shared style).
- `fluctuationofmagneticfiel.py`, `plot_prt.py` and `validate_moments.py` moved
  to `plot_style` (they had a dark theme, 150–220 dpi, no PDF, no check); the
  Okabe-Ito palette replaced a red/green pair that colour-blind readers cannot
  separate. ~300 lines of unreachable or never-called code removed.
- One convention for every map (z along B0 horizontal, equal aspect, integer
  d_i ticks); `structures_analysis` and `heat_flux_analysis` drew the
  transpose. k_par horizontal in all k-space maps (the mirror dB_z spectrum
  was transposed). Time as t Omega_ci everywhere, velocities as v/v_A.
- Axis offsets folded into labels, log axes with 1-2-5 labels, autocrop
  clamped to the computed grid, headroom above extreme points, legends moved
  off data; every one of the 88 figure types was inspected visually.

### Physics corrections found in the review

- `plot_prt.py` and `vdf_spatial.py` binned u = gamma v while labelling v; both
  now convert, weight and normalise by the whole population.
- The |v_perp| reference was a 1-D Gaussian; it is the Rayleigh distribution of
  a 2-D Maxwellian (verified on the synthetic bi-Maxwellian).
- `mirror_area_fraction` (cells below B0 - std|B|) is ~16–25 % for any amplitude:
  renamed and no longer shown as a hole area.
- The spectral index was fitted over the middle half of k regardless of the
  physics and reported a noise slope k^+2.8; it is now fitted only over a
  decaying range above the noise floor (known-answer test added).
- kappa fits at their bound (80) are shown as Maxwellian-consistent in 1/kappa
  with conditional errors; the A-vs-field scatter files were overwritten at
  every step (only the last survived) and plotted raw |B| in code units.
- A factor-2 labelling error in the magnetic-energy panel of `plot_prt.py`.

`METHODOLOGY_REVIEW.md` evaluates every method physically. Its main findings:
the moderate "mirror" runs are equally unstable to the ion-cyclotron branch,
and the strongest v5 mode is parallel and transverse (IC-like), so mirror
growth must be measured on the oblique compressive component; electrons carry
up to half of the perpendicular pressure by the end because of the numerical
heating; the 20 d_i box samples only k rho_i = 0.70, 1.40, 2.11; a single
realisation per case cannot yet establish a distribution effect.

Verification: 205 tests and 14 subtests pass without warnings; the full
pipeline on the synthetic run writes 147 figures with 0 content or layout
findings; no unused imports remain; the COSMA job was rerun locally end to end.

## Follow-up 3 (2026-09-29): smaller control box and resolution test

- The isotropic controls now run in a 10 d_i box at 288² (`PSC_DOMAIN_DI 10.0`,
  `PSC_NGRID=288`, 256 ranks on 10 nodes): the twins' dx, dt, ppc and cadence
  with a quarter of the cells. The energy audit pairs controls by dx, ppc and
  electrons rather than by box, and subtracts their energy changes per unit
  volume (ratio of E_B(0)); known-answer test with a quarter-volume control.
  The case-integrity rule for `PSC_DOMAIN_DI` records the exception.
- Resolution test of `psc_mirror_bimaxwellian_moderate` to t Omega_ci ~ 41:
  `PSC_NICELL=4000` and `PSC_NGRID=1152` (dx/lambda_De 4.3, dt/2), same
  executable, production cadence in physical time. Each job writes
  `analysis_config.json`, which `run_pipeline.py` now uses automatically, so
  the manifests carry the real ppc and dt (`runtime_verified`).
  `analisis_resolution_mirror.sh` runs both variants, the energy audit with
  labelled runs and a comparison at the last common time, and the convergence
  table; the whole chain was run locally on synthetic variants.

## Follow-up 4 (2026-09-29): mode identification

Why: the three weakest links left between the measurements and the thesis
claims were (i) one growth rate per run although mirror and ion-cyclotron
compete at the same anisotropy, (ii) a psi_pm handedness taken from a sign
convention, and (iii) Brazil thresholds for cold electrons in runs whose
electrons heat to beta_e ~ 8.

- Branch-resolved growth: `mode_power(..., split=True)` gives the transverse
  and compressive power of each followed mode; `classify_mode` assigns
  compressive-oblique (mirror) or transverse-parallel (IC) from theta_kB and
  compressibility on the mode's own linear phase; candidates include the
  strongest modes of each part so a weaker branch is always followed. New
  series `mode_compressive` / `mode_transverse`, `branches` in
  `linear_phase.json`, `branch_growth` QA check, per-branch comparison figure.
  Known answer: synthetic run with a mirror mode (gamma 0.25) and a left-hand
  IC wave (gamma 0.15, `synthetic_run.py --ion-cyclotron`) returns 0.236 and
  0.142, compressibility 0.90 and 4e-13.
- Handedness: `handedness_check` integrates the ion gyration (Boris) and
  verifies, through `stream_polarization` and `temporal_dispersion`, that
  left-hand waves land on psi_+ and right-hand ones on psi_- at omega > 0,
  for both propagation directions and both signs of B0 (psi_pm now include
  sign(B0)); a flipped time kernel is detected. On the synthetic run the IC
  wave appears in psi_+ at (k d_i, omega/Omega_ci) = (0.63, 0.46).
- Thresholds: `mirror_criterion` / `mirror_threshold_electrons` (Hellinger
  2007, Eq. 16, protons + electrons); `anisotropy_analysis.py` reads the
  electron moments of every snapshot and draws the threshold of the measured
  (beta_e||, A_e); CSV columns `beta_e_parallel_global`,
  `anisotropy_e_global`, `marginal_threshold_cold_electrons`.
- Tests: `test_branches_handedness.py` (15 tests).

## Follow-up 5 (2026-09-30): what the v6b mirror results required

Why: the v6b reanalysis showed that the "mirror" runs are dominated by the
parallel ion-cyclotron instability (k_par d_i = 0.314, k_perp = 0,
compressibility ~1e-11, left-hand), with growth rates within 2 % of the
parallel kinetic theory (0.1226 / 0.1191 / 0.1114 vs 0.1236 / 0.1194 /
0.1134 Omega_ci for Maxwellian / kappa 5 / kappa 3), and no measurable mirror
growth. The pipeline was adapted to that result.

- Mode fits start after the quiet-start noise settles, 2 / (k v_th,i): weak
  modes had been accepted at gamma ~ 2-6 Omega_ci on [0.13, 0.3] Omega_ci^-1.
- `theory` runs for mirror cases and writes the competing parallel IC branch
  ('plus' channel only); the runner no longer refuses it and the COSMA job
  runs it for mirror_*. Polarization maps and growth tables get gamma_theory.
- omega-k maps are shown in |k| <= 2, |omega| <= 2 (ion units) with the
  theory at +-k, unstable band bold and damped part thin; the full transform
  spanned |k d_i| ~ 90 and the modes were a pixel at the origin.
- `mode_amplitude_timeseries.csv`: amplitude and compressive fraction of the
  dominant mode, the branch leaders and the ten strongest modes.
- `vdf_spatial.py` measures the local-field kappa_eff at 24 snapshots
  (was 6); `kappa_evolution.py` documents that its global-B0 fit reads the
  wave-tilted distribution at saturation as a tail.
- Log-scale fluctuation plots skip the t = 0 uniform field
  (`plot_style.measured_fluctuation`).
- `paper_figures.py` builds the series figures from analysis products only.
- Legends and annotations moved off the data in every figure the QA flagged
  on the v6b runs (Brazil, anisotropy evolution, energy partitions, trapping
  VDFs, comparisons, kappa evolution).

## Follow-up 6 (2026-09-30): pre-launch audit of the final analysis (v6c)

Why: v6c is the last reanalysis before the paper; every change was checked
on the real v6b products and on a synthetic end-to-end run of the COSMA job.

- Growth rates of non-dominant modes are fitted only before the dominant mode
  saturates: afterwards the anisotropy has relaxed and the saturated wave
  drives other k nonlinearly, so the rate is not the linear growth of the
  initial state (v6b kappa = 5: a compressive mode "grew" at 0.25 Omega_ci
  on [58, 66], after the ion-cyclotron wave saturated at 49).
- energy_exchange.py: the snapshot integral of J_s.E is not the work when the
  output interval aliases the plasma oscillation of the species (v6b kappa 5,
  electrons: cadence * omega_pe = 165, int <J_e.E> dt = -0.72 vs Delta K_e =
  +0.036 from DiagEnergies). The closure is graded per species
  (PASS <= 10 %, FAIL >= 50 %), aliasing is reported, and an aliased,
  unconfirmed curve is drawn dotted, labelled, and kept out of the scale.
- Physical units instead of code units: T_i / (m_i v_A^2), energy densities
  / (B0^2/mu0), J_dia / (n0 e v_A), the energy proxy relative to t = 0.
- COSMA job: default tree `analysis_results/v6c`; isotropic controls skip the
  theory stages (no drive; a failed theory stage would hold back their
  spectral stage); mirror series get `paper_figures.py`. The resolution
  analysis compares against v6c and includes the theory stage.
- linear_theory.py: a kappa seed dragged from the marginal root of the first
  k (gamma ~ 1e-18) sat on the Im(omega) = 0 boundary where Z_kappa is not
  integrated, and the kappa = 5 ion-cyclotron branch was lost for every k.
  The seed is lifted to Im(omega) >= 1e-3; kappa 5 / 3 now give gamma_max =
  0.1225 / 0.1162 at k d_i = 0.36 (regression test). Damped kappa roots
  (gamma < 0) remain out of reach of this Z_kappa; figures draw only what is
  found.
- Figures that the preflight showed empty or crowded for a stable plasma: the
  growth-rate map says "no mode with an accepted growth fit", the kappa
  panel of vdf_b_profiles says "no resolvable tail", round-off axis spans
  get a +-1 % window, corner tick labels are dropped.
- The job summary lists the runs it did not analyse and why (unfinished,
  missing, already present), and `find` follows symlinked run folders.
- Preflight: the COSMA job script itself was run on a laptop (stand-in srun,
  local paths) on four synthetic runs and an isotropic control: every stage
  PASS, controls without theory stages, comparison, kappa evolution, paper
  figures and evidence report OK, no figure-QA issue that applies to real
  data.

## Follow-up 7 (2026-09-30): figure pass for the thesis and the paper

Why: the v6b figures were reviewed one by one for publication; the VDF
figures in particular misled (a kappa model that looked wrongly normalised),
and several diagnostics showed mostly noise or empty axis range.

- **Kappa normalisation.** The bi-kappa formula was right: PSC loads a
  Student t with nu = 2 kappa - 1 and a shared mixing variable, whose 1D
  marginal (1 + v^2/((2 kappa - 3) sigma^2))^-kappa has variance sigma^2;
  `test_vdf_models.py` checks the unit integral, the variance, the 3D, reduced
  and speed densities, and the Poisson agreement with a sample drawn like the
  loader. What was wrong was the comparison: the last snapshot against kappa0
  at the *initial* temperature, off by the heating factor. Every model is now
  drawn with the variance measured at the same time, and kappa is estimated by
  maximum likelihood with that variance fixed (`plasma_physics.kappa_mle`,
  bootstrap 68 % interval, unbiased on PSC-like samples; the least-squares fit
  of the binned density gave 2.24 for a loaded kappa = 3). The t = 0 column of
  `kappa_comparison_*` checks the loader. `synthetic_run.py` now draws its
  bi-kappa particles the way PSC does.
- **VDF figures redesigned.** plot_prt.py: `kappa_comparison_{parallel,
  perpendicular}` (t = 0 and final, Poisson error bars, Maxwellian / MLE kappa
  / kappa0, ratio panels), `goodness_of_fit` (F_PIC - F_model with the 95 % KS
  band, KS and AD in a CSV), `distribution_change_{ions,electrons}`
  (log10 f(v,t)/f(v,0)), `vdf_1d_evolution`. physical_diagnostics.py:
  `vdf_2d_*` per d^3v with equal axes, a 5-sigma window, masked low-count bins,
  contours at 1e-1..1e-3 of the peak and the initial model at the same levels
  (white contours carry a dark rim, visible in the legend too), an .npz of each
  histogram; `vdf_planes_*` replaces the unreadable 3D scatter;
  `kappa_vs_maxwellian_*` in v/v_A with a ratio panel; `kappa_fit_vs_time`
  with MLE error bars. paper_figures.py adds `vdf_evolution`: f(v||, v_perp)
  of every run at t = 0, the end of the linear phase and the end, with the
  initial model and the cyclotron-resonant velocities.
- **Growth curves start after the quiet-start noise build-up.** t = 0 (uniform
  field) and t < 2/(k v_th,i) (psc_units.noise_settling_time) are no longer
  drawn in `growth_curve_*`, `mode_growth_*`, `growth_rate_fit_*` and
  `mode_amplitude`; the hidden interval is stated on the figure.
- **Dispersion diagrams.** The pipeline runs the generic preset, which had no
  k cap: the omega-k diagram of v6b ran to k d_i = 40 with the ion-cyclotron
  ridge in one corner, and grid noise striped the phase-velocity projection.
  `--kmax-di` now defaults to psc_units.K_MAX_DI_DEFAULT (k d_i <= 2 ion-scale,
  k d_e <= 2 electron-scale; never fewer than two modes per axis), and the
  omega-k and mode-summary panels display that band.
- Units and labels: diamagnetic current maps in n0 e v_A (were code units);
  the heat-flux block maps use the truncated estimator (the untruncated third
  moment of a kappa = 3 sample has infinite variance and one block set the
  colour scale), a robust colour scale, and hatch the blocks within 2 sigma of
  the symmetric null; the field-residual title no longer announces an abort
  threshold that is not drawn; identically zero fluctuation maps (t = 0) are
  not saved; one legend below the estimator panels; the helicity figure notes
  that counter-propagating waves of the same polarization cancel in sigma_m;
  the energy proxy says what it contains.

## Follow-up 8 (2026-09-30): the kappa index against the magnetic field

Question from the review: how does kappa change in time, and is the change
organised by the magnetic field at the same instant?

- **What the v6b products already show** (local-field frame, truncated
  whitened kappa_eff, 6 snapshots): kappa 3.01 -> 4.81 and 5.03 -> 8.92 by
  t Omega_ci = 158; the bi-Maxwellian run stays Maxwellian. The index is the
  same in |B| holes, ambient plasma and peaks and flat in b = |B|/B_ref at
  every time (slope d(1/kappa)/d ln b consistent with 0; hole - peak within
  +-5e-3): an ion crosses the 4 d_i window in ~2.5 Omega_ci^-1 at the thermal
  speed, against an e-folding time of 1/kappa of ~280-340 Omega_ci^-1, so no
  spatial structure in kappa can survive; kappa is a property of the whole
  population, not of the local field strength. In time, ln[(1/kappa)/(1/kappa_0)]
  = -nu_0 t - c int_0^t W dt with W = <|dB|^2>/B0^2 fits both runs (R^2 >
  0.997): c = 0.069 +- 0.007 (kappa 3) and 0.063 +- 0.007 (kappa 5), the same
  within errors (one c for both runs: 0.066 +- 0.005, chi2/dof 4.7 against
  4.7 and 5.8 for the separate fits), and nu_0 = (1.45 +- 0.11)e-3 and
  (2.17 +- 0.11)e-3 Omega_ci. About half of the final decrement is the wave term; the other half
  acts from t = 0, before the waves grow, as the numerical relaxation of a PIC
  plasma would. The isotropic controls with the same kappa measure nu_0
  without waves and decide it.
- `kappa_dynamics.py` (new): that analysis from products, three figures and
  the tables; controls are paired automatically in the same tree. Run by the
  COSMA job for every series; a later job that analyses a control re-runs the
  comparisons of its series. paper_figures.py `kappa_local` is its first
  figure.
- vdf_spatial.py writes the index at *every* particle snapshot
  (`vdf_kappa_series.csv`, `vdf_kappa_b_series.csv`; ~120 points instead of
  24), without figures and with the influence-function error of the truncated
  kurtosis (kappa_eff._K_standard_error; Monte Carlo: equal to the scatter of
  independent samples within 10 %, like the bootstrap).
- kappa_eff.signed_inverse_kappa: 1/kappa continued through the Maxwellian
  (negative = flatter than Maxwellian, e.g. a resonant plateau), so the
  bi-Maxwellian runs get a value with an error instead of "inf", and the
  index is continuous on one axis for all runs.
- kappa_evolution.py draws the local-field index as the main curve and the
  global-B0 fit faintly. The fit's bump at t Omega_ci ~ 60-90 (present even in
  the bi-Maxwellian run, kappa_fit ~ 14-16) is not a tail: there the kappa
  model fits the tail *worse* than the Maxwellian (tail error 0.39 against
  0.20), and the fraction of ions beyond 3 sigma, 0.27 % for a Gaussian,
  *drops* to 0.15 % and recovers to 0.25 % by the end. Right after
  saturation the ion-cyclotron wave flattens the parallel distribution
  (resonant diffusion), a transient the kappa fit misreads; the field tilt is
  not the cause (A in the local and in the global frame agree at t = 63). In
  the kappa runs the tail content only decreases, fastest around saturation.
  kappa_field_evolution.png now shows this model-free tail content between
  the index and the field energy (vdf_kappa_series.csv carries it in the
  local frame; fit_metrics.csv in the global one for older deliveries).
- **The shape figure** (`kappa_shape_evolution.png`, kappa_dynamics.py):
  f(v_par) in units of its own sigma at every snapshot (vdf_spatial.py writes
  the standardised histograms, `vdf_shape_series.npz`, local-field frame),
  as the change since t = 0 in time (map) and against the Gaussian of the
  same variance at t = 0, the end of the linear phase, 30 Omega_ci^-1 later
  and the end, with the n = 1 cyclotron resonance of the measured mode and
  the reference shapes of a kappa tail (wings above 1) and of a flattening
  (shoulders above 1, core and wings below). Dividing by sigma(t) removes
  the heating and the anisotropy relaxation, so the figure shows the shape
  alone: it is the direct test of "flattening at saturation, not a tail".
  `kappa_shape_metrics.csv`: core (|u| < 0.5), shoulder (1 < |u| < 2) and tail
  (|u| > 3) probability over the Gaussian at every snapshot.
