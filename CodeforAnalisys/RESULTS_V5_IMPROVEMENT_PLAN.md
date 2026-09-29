# Results v5: evidence-based analysis improvement plan

Date: 2026-09-28. Scope: Python post-processing and the local `results/v5` delivery.
Status: planning and audit only; no processing or simulation code changed.

## 1. Scope and governing conventions

The local delivery contains three mirror-moderate runs (bi-Maxwellian, kappa 5 and kappa 3), their comparison and kappa evolution: 884 PNGs, 674 PDFs, 15 GIFs, 149 CSVs and 35 JSONs. This audit inventories the delivery, examines key machine-readable diagnostics and visually inspects eight representative figures; it is not an inspection of every image. Raw production snapshots and original COSMA logs were not available for reprocessing in this review.

Follow `CLAUDE.md`, `.claude/skills/psc-case-integrity/SKILL.md`, `.claude/skills/graphify/SKILL.md`, `CodeforAnalisys/README.md` and the existing physical audit. The knowledge graph was queried using its installed executable; source and result files take precedence over potentially stale graph locations. New documentation and figure labels stay in English, and plotting uses `plot_style.py`.

Declared shared parameters are beta_i_parallel=5, A_i=2, beta_e_parallel=1, A_e=1, mass ratio=200, B0=0.08, domain approximately 20 d_i, grid 576x576 and nominal nicell=1000. Each manifest records 2401 field/moment pairs and 121 particle snapshots through step 1,200,000. The manifests explicitly identify dt as a CFL estimate and nicell as a profile value requiring runtime verification. These are not fully certified runtime parameters.

Changing grid, timestep, particles per cell, box or output cadence belongs to a controlled convergence campaign, preserving distribution twins. It is not a post-processing fix. Do not alter production parameters to make an analysis pass. Preserve the existing local edits in ten analysis/documentation/test files; several proposed remedies are already implemented there and require validation and regeneration.

## 2. Observed evidence and its implications

| Evidence in local v5 | Observation | Consequence |
|---|---|---|
| `comparison_mirror_moderate_kappa/comparison_anisotropy.png` | Legend and axes but no visible curves | Cadence/NaN plotting defect; local comparison edits already address finite samples |
| `*/09_physical_diagnostics/growth_rate_summary.csv` | Reference is still `total`; no `mode` row | Existing figures predate the local modal-growth correction |
| Same files | Old total gamma/Omega_ci: Maxwellian 0.06470; kappa 3 0.05144; kappa 5 0.05436 with `fit_ok=0` | Do not publish a distribution ranking using these stale estimators; rejected kappa 5 is not zero growth |
| Parallel growth fits | Windows near t Omega_ci=0.26–4.42, all rejected | Early transient mistaken for a physical fitting window |
| Kappa 3 growth-map JSON | Window 15.83–95.00 of a 0–158.33 run | Old fixed fraction can span different physical stages |
| All three dispersion-mode JSONs | `dominant_mode_not_confirmed`; strongest peak k_parallel d_i=0.31416, k_perp=0; 96.3–97.5% retained power; compressibility near 10^-12 or less | Case label does not establish mirror-mode identification; audit coordinates/components and competing modes before interpretation |
| Kappa 5 global-energy JSON/CSV | E_total: 3837.271 to 6314.394; relative change +0.64554; error/exchanged energy=0.84976 | Urgent validation issue. Verify original diagnostic and its producer; do not attribute this to a specific numerical mechanism yet |
| Maxwellian and kappa 3 energy exchange | `diag_available=false`; no local global-energy summary | Global conservation is unverified for those deliveries |
| Kappa 5 energy exchange | `diag_available=true`, but final dK and mismatch are NaN | Availability is not successful energy closure |
| Field residuals | Maxwellian/kappa 3: no Gauss/continuity log checks; no thresholds | Missing evidence must not be reported as a pass |
| Structure summaries | No holes under current definition; final pressure-balance ratio 0.747/0.985/1.037 for Maxwellian/kappa 5/kappa 3 | Density anticorrelation alone does not establish pressure-balanced mirror structures |
| Initial validation | All three initial particle summaries PASS; particle window 13225/331776 cells (~3.99%) | Useful initialization evidence, but particle-window conclusions are not automatically domain-wide |
| Reanalysis job summary | Stages report OK despite the above | Process success and scientific validity need separate statuses |

The kappa 5 timing mismatch has a concrete candidate cause: the last energy time is 395833.0, while step times reconstructed from the estimated dt reach approximately 395833.316. `energy_exchange.py` interpolates outside the diagnostic range to NaN. Verify actual runtime dt and output rounding, then match with a documented tolerance; do not silently extrapolate physical data. This explains a possible endpoint NaN, not the large energy increase or any interior closure discrepancy.

## 3. Priority P0: establish trustworthy measurements

### P0.1 — Provenance, runtime parameters and scientific status

Files: `write_analysis_manifest.py`, `run_geometry.py`, `psc_units.py`, `compare_physical_cases.py`, `Makefile`, reanalysis job script.

Record code commit, dirty-tree/source hash, analysis schema and algorithm versions, CLI/configuration, package versions, input identity, runtime dt/nicell, output cadence, geometry and parameter sources. Give each stage separate execution and scientific statuses: PASS, WARN, FAIL, UNVERIFIED, with reasons. Use strict JSON (`null` plus reason for unavailable values, not NaN). Reject comparisons mixing growth estimators or incompatible algorithm versions; retain old results in a separate directory.

Acceptance: a missing log, failed growth fit or missing energy budget cannot yield an unqualified validated comparison. Equal profiles do not substitute for measured initial-state and runtime checks. Produce a three-case validation matrix linking every decision to its source.

### P0.2 — Resolve the energy anomaly before interpreting saturation

Files: `energy_conservation.py`, `energy_exchange.py`, `physical_diagnostics.py`, `field_residuals.py`, PSC `DiagEnergies` producer headers for read-only verification.

Retrieve original `diag.asc`, run logs and build identity for all three cases. Verify species ordering, field factor 1/2, particle weights, volume, restart segments, time units and the actual producer revision. Cross-check selected snapshots against independently reconstructed energies, explicitly distinguishing the particle subwindow from the full domain. Plot energy changes by reservoir, total drift, drift relative to exchanged energy and drift during the proposed linear interval. Investigate the strong electron-energy rise without detrending it away.

Align energy and J_s dot E using verified times and bounded rounding tolerance; report overlap coverage and the last mutually supported time. Audit the fallback current q p/m (u versus v convention), spatial Yee co-location, temporal staggering and output-cadence aliasing. A finite endpoint is insufficient: inspect the full cumulative closure residual.

Acceptance: a documented explanation or a scientific FAIL/UNVERIFIED status. Proposed tolerances must be justified against the size of the physics signal and convergence evidence, not chosen after seeing results. If the anomaly persists, post-processing cannot repair the trajectory: plan controlled reruns separately. Early-time results can only be retained with an explicit early-time error assessment.

### P0.3 — Finish validating the existing growth-rate corrections

Files: `growth_fit.py`, `physical_diagnostics.py`, `growth_rate_map.py`, `spectral_analysis.py`, `polarization_dispersion.py`, `dispersion_modes.py`, and their consumers.

The working tree already adds modal growth, a transient-resistant window search and modal-reference consumption. Review and test those edits rather than duplicate them. Distinguish strongest power, fastest accepted growth and a physically classified mode. The current dominant choice based on maximum amplitude over the run needs a sensitivity comparison against a linear-phase ranking: a late nonlinear mode need not be the linear instability.

Persist mode indices, physical k, amplitude/power convention, fitting interval, gain, R-squared, rejection reason, noise estimate and uncertainty components. Use one stored run-phase definition where meaningful, but retain mode-specific diagnostics when branches grow at different times. Never lower acceptance criteria solely to obtain a kappa 5 bar. A common fitting routine does not by itself guarantee identical fitting windows across scripts.

Acceptance: known-growth synthetic signals recover gamma with noise, an initial transient and saturation; power fits use the required factor of two; no-growth signals remain rejected; consumers consistently read the accepted reference or expose its rejection. Regenerate physics, spectrum, growth maps, polarization, dispersion and dependent comparisons/kappa/convergence products together.

### P0.4 — Confirm what mode grew

Files: `run_geometry.py`, `streaming_fields.py`, `dispersion_modes.py`, `dispersion_analysis.py`, `spectral_analysis.py`, `polarization_dispersion.py`.

Verify dimension ordering, parallel direction, magnetic-component mapping, conjugate-pair treatment, background subtraction and FFT normalization against a known injected oblique and parallel wave. For measured candidates, jointly report angle, compressibility, polarization/helicity, amplitude, coherence and frequency uncertainty during the linear phase. Keep the possibility of competing branches open. Separate full-run power ranking from phase-specific characterization.

For a near-aperiodic candidate, distinguish a bounded frequency consistent with zero from a failed/incoherent frequency estimate. Do not require artificial propagation, and do not call every unresolved frequency zero. Keep the full spectral survey but add an ion-scale zoom; the inspected full-range panel hides k~0.314 near the origin.

Acceptance: an evidence-based classification or an explicit unclassified mode, independent of `CASE`. Do not compare a parallel branch against a mirror growth prediction.

## 4. Priority P1: strengthen inference from existing data

| Work package | Files | Proposed improvement and acceptance |
|---|---|---|
| Comparison plots | `compare_physical_cases.py`, `kappa_evolution.py` | Validate local finite-sample fix; preserve real gaps and original cadence; missing/invalid fits shown as unavailable without zero-valued error bars. Export common time coverage, fit status and estimator identity |
| Spectral maps | `growth_rate_map.py`, `spectral_analysis.py` | Display calculated/accepted/rejected regions separately; choose limits from retained k or label wider uncomputed regions; show discrete bins and Delta k=2pi/L (~0.314 d_i^-1). No smoothing that suggests extra resolved modes |
| Uncertainty | `growth_fit.py`, `convergence_study.py` | Separate fit error, window sensitivity, correlated time-sample uncertainty and independent-run spread. Assess block resampling or a correlated-residual model on synthetic data; snapshots are not independent realizations |
| Thermodynamic estimators | `plasma_physics.py`, `validate_moments.py`, `estimator_consistency.py`, `anisotropy_analysis.py` | Compare particle and moment estimates on the same spatial window and time; distinguish mean(T_perp/T_parallel) from ratio of averaged pressures, global/local field bases and u-based initialization from later pressure definitions. Export effective sample size and mismatch reasons |
| Kappa and VDFs | `kappa_eff.py`, `vdf_spatial.py`, `physical_diagnostics.py`, `kappa_evolution.py` | Compare fitted and moment-based kappa; uncertainty and upper-bound/unidentifiable status near Maxwellian; sensitivity to truncation, bins, macrocells and weights. Show fit residuals and distinguish core from tail evolution. Verify sampling-window representativeness |
| Heat flux | `heat_flux_analysis.py` | Retain third-moment and truncation implementation. Calibrate a kappa- and truncation-matched null/noise floor, since the inspected plot uses a Maxwellian reference. Report signed and absolute flux, blocks, particle counts and truncation sensitivity; validate zero flux in symmetric/drifting controls |
| Structures | `structures_analysis.py` | Report both change relative to B0 and fluctuation relative to the instantaneous mean/background, so a mean magnitude increase is not automatically a population of peaks. Distinguish |B| effects from delta B_parallel. Sweep threshold and smoothing, report filling fraction, size distributions and pressure-balance uncertainty |
| Pressure balance and geometry | `structures_analysis.py`, shared field utilities | Co-locate fields and thermal moments; compare smoothing B squared with squaring smoothed B (currently the latter), quantify their effect. Test periodic connected structures and geometry before assuming arbitrary planes are supported |
| Structure evolution | `structures_analysis.py`, `vdf_spatial.py` | Track individual structures through periodic boundaries with split/merge flags. Current interval above half maximum area fraction is a population activity duration, not an individual lifetime. Pair local VDFs with verified structures at equal phases |
| Diamagnetic current | `diamagnetic_current.py`, `physical_diagnostics.py`, `plasma_physics.py` | Keep existing shared sign/thermal-pressure fix. Add measured-current comparison and gradient-resolution sensitivity with consistent co-location; do not equate one current contribution with the full deposited current |
| Theory | `linear_theory.py`, theory-import consumers | The existing solver explicitly excludes mirror and only treats parallel propagation. Define an interface for independently validated oblique theory with matching distribution/temperature conventions; report solver residuals and interpolate onto actual discrete simulation modes. External solver selection and literature validation are separate research work |

## 5. Priority P2: reproducibility, performance and publication

1. **A controlled convergence matrix.** Once post-processing is validated, vary dx, dt, ppc, domain and random seed separately; preserve physical distribution twins at each setting. Use `convergence_study.py` for gamma, dominant k, saturation, energy drift and selected structure/VDF observables. A larger box tests accessible wavelengths, not just visual smoothness. Record run identity and independent realizations; do not concatenate independent restarts.
2. **Shared data primitives.** Consolidate geometry, centering, units, selection, weights and finite-value handling without rewriting the pipeline at once. Make raw-data loading separate from diagnostics and plotting. Prioritize correctness tests before refactoring `physical_diagnostics.py`.
3. **Dependency-aware execution.** Express true prerequisites in the Makefile/job graph; listing prerequisites does not ensure scientific order under parallel make. Atomic stage outputs, incomplete/failed-stage markers and selective regeneration should prevent mixed old/new products. Avoid unconditional cleanup as part of a planning task.
4. **Measured I/O optimization.** Profile representative field and particle stages first. Cache compact derived arrays with input/configuration hashes, stream snapshots, reuse selected Fourier coefficients and avoid nested process/thread oversubscription. Benchmark peak memory, elapsed time and output equivalence; do not cache full runs indiscriminately.
5. **Figure selection.** Retain the complete diagnostic archive but curate a compact thesis set: initial-state validation; energy/residuals; mode identification and gamma(k); anisotropy/field evolution; structures/pressure balance; local VDF/kappa. Add heat flux or energy transfer only where validated. Use consistent distribution colors, aligned axes, symmetric signed color maps, shared comparison scales, shorter titles and explicit units/status/window.
6. **Automated report and visual checks.** Build an index linking figures, CSVs, configuration and validation reasons. Detect empty series, all-NaN panels, missing expected files, clipped annotations and inconsistent legends. A successful figure save is not sufficient. Compare static exports at their intended thesis size.
7. **Documentation repair.** Correct stale comments and distinguish historical v5 from the revised algorithm. The parity checker currently reports 18 pre-existing findings: stale grid/ppc comments and two monolithic reconnection cases outside its parity checks. Address these in a separate documentation/migration work item, not as a hidden parameter change.

## 6. Execution sequence and deliverables

| Batch | Dependency | Deliverable | Exit condition |
|---|---|---|---|
| A: inventory and provenance | Existing delivery | Validation matrix, runtime-source inventory and frozen baseline | Every unknown is visible; existing local edits preserved |
| B: energy and modal audit | A plus original diagnostics/logs | Energy verification, time-alignment report, classified or unclassified modes | No unsupported conservation or mirror-mode claim |
| C: validate local corrections | A; B informs interpretation | Focused tests and synthetic end-to-end outputs | Growth/NaN regressions pass; versions recorded |
| D: production regeneration | B/C, raw snapshots on COSMA | Separate revised results directory and refreshed dependent comparisons | Uniform algorithm versions and explicit scientific statuses |
| E: uncertainty and nonlinear diagnostics | D | Robust VDF, heat-flux and structure results with sensitivity tables | Conclusions remain stable across justified estimator choices |
| F: convergence and theory | B–E | Controlled new-run proposal and validated theory comparison | Numerical/realization uncertainty smaller than the claimed distribution effect, or an explicit unresolved result |
| G: thesis figures | Validated subsets from D–F | Curated figures, tables and traceable report | Every plotted claim links to accepted evidence |

The fastest useful first implementation is provenance/status reporting, validation of the existing growth/comparison edits, and the energy/time-alignment audit. More plots should follow those checks, not substitute for them. New simulations require a separate controlled specification; this plan does not submit jobs or change physical parameters.

## 7. Review limits and checks

Eight images were visually inspected: comparison anisotropy, comparison growth, kappa 5 partial-energy change, and kappa 3 parallel growth fit, dispersion modes, structure map, parallel growth map and heat flux. Numeric conclusions above come from JSON/CSV evidence, not digitized plots. The reported 64.6% change is independently present in the global-energy summary, not inferred solely from the partial-energy proxy figure.

The case-parity checker was run read-only and returned 18 pre-existing findings. Focused test results are recorded below. Production reruns, full-suite validation, external-theory validation and exhaustive image QA remain work packages, not completed actions.

Focused verification: `.venv/bin/python -m pytest -q CodeforAnalisys/test_growth_fit.py CodeforAnalisys/test_growth_rate_map.py CodeforAnalisys/test_energy_conservation.py` completed with **14 passed and 12 subtests passed** in 18.56 s. These tests validate selected code paths in the current working tree; they do not certify the production trajectories or regenerate the existing figures.

## 8. Follow-up: additional blind spots found in source review

These additions distinguish observed implementation differences from proposed diagnostics. Their numerical impact on the production results remains to be measured.

### A1 — VDF coordinate semantics: u is not v (P0)

In `physical_diagnostics.py`, `fit_distribution()` takes `snapshot.pz` directly, and the 2D VDF takes `snapshot.px/py/pz` directly while labelling axes as velocity. The reader returns PSC momentum-coordinate data; `particle_temperatures()`, `particle_heat_flux()` and `particle_energy()` explicitly call `velocity_from_u()`, whereas these VDF paths do not. Audit the 3D rendering path as well.

Choose explicitly whether a diagnostic estimates f(u) or f(v). Convert particle samples before a velocity histogram, or label momentum-coordinate products correctly; a fitted initialization model in u cannot automatically be reused as the same analytic model in v. Quantify the difference for electrons and high-energy tails at early and late times. Acceptance: a synthetic sample with appreciable u/v differences is plotted and fitted in the declared coordinate, and the nonrelativistic limit agrees. Do not assume a small initial bulk correction stays small during the observed electron-energy rise.

### A2 — A cylindrical histogram is not the same quantity as a gyrotropic VDF (P1)

The integrated 2D plot uses `histogram2d(v_parallel, v_perp, density=True)` and labels it f(v_parallel,v_perp). `vdf_spatial.py` instead divides weighted bin populations by the cylindrical velocity-space volume. Both representations can be useful, but they answer different questions and should not share an ambiguous label or be compared as the same density.

Provide explicit choices: probability per dv_parallel dv_perp, and gyrotropic phase-space density per velocity-space volume. Use exact annular bin volumes near v_perp=0, document normalization, and export the probability excluded by percentile cuts. The current `density=True` normalization is conditional on the displayed range. Acceptance: integrating each representation with its own measure recovers the documented probability; a synthetic Maxwellian does not acquire an interpreted physical ring solely from the radial measure.

### A3 — Solver fallback must preserve the statistical objective (P1)

`fit_distribution()` calls `curve_fit` without sigma, fitting unweighted residuals in density space. `_grid_fit_distribution()` selects parameters using log-density error, with amplitudes separately fitted in linear space. Therefore dependency availability or an optimizer exception can change the scientific estimator, rather than merely the numerical method. The broad exception handler also obscures why fallback occurred.

Specify one objective and noise/weight model; implement the same criterion in both paths or explicitly expose them as different estimators. Record solver, objective, convergence/boundary flags and fallback reason. Separate weighted histogram content from raw counts/effective sample size when applying minimum-bin thresholds. Validate both paths on identical synthetic distributions, including sparse tails and unequal weights.

### A4 — Compare models without automatically rewarding the extra parameter (P1)

Kappa has an additional shape parameter relative to the fitted Maxwellian. A smaller in-sample residual alone is insufficient evidence that the tail is physically kappa-like. Define held-out predictive checks or an explicitly justified likelihood/model-selection procedure; include the near-Maxwellian boundary and uncertainty. Add a mixture-of-local-Maxwellians control to test whether an apparent global tail can arise from spatially varying temperatures/drifts. This complements, rather than replaces, local VDF analysis.

Acceptance: the procedure controls false kappa detections on Maxwellian controls and distinguishes a true kappa sample from a spatial mixture to the extent permitted by available data. Report ambiguity when it cannot.

### A5 — Candidate discovery can miss an early-lived mode (P1)

`mode_candidates()` retains eight strongest modes from snapshots at 25%, 50%, 75% and 100% of the run. This is efficient but does not guarantee capturing a mode that grows and disappears before the first candidate snapshot. It also differs from the already identified issue of ranking candidates by their late amplitude.

Add onset-aware or logarithmically spaced early candidate snapshots and a bounded streaming candidate union. Compare the retained set against a denser pilot scan. Test a two-mode synthetic trajectory where the early mode disappears before one quarter of the run; its growth must remain discoverable. Report candidate coverage and selection sensitivity.

### A6 — Make subsampling independent of execution order (P2)

The integrated diagnostics use a module-global seeded RNG; heat-flux processing passes a sequential seeded RNG through the loop. A fixed seed alone does not establish invariance to selected snapshots, species processing order or multiprocessing arrangement.

Derive deterministic sampling seeds from stable run identity, step, species and diagnostic purpose, and record the sampling scheme. Test one-step versus full-run execution and serial versus parallel execution. Keep statistical sampling randomness separate from the physical random seed of the PIC run. Acceptance: the same diagnostic request reproduces its sample independently of scheduling, or documented statistical equivalence is used where exact identity is impractical.

### A7 — Normalized averages and physical averages must be distinguished (P1)

`heat_flux_analysis.window_mean()` particle-weights already normalized block values q/q0. This is not generally the same as a volume-averaged dimensional heat flux divided by a domain reference q0. State which estimator is plotted and expose both when the scientific question needs transport rather than a typical particle-weighted local ratio. Test blocks with different density and temperature; do not silently compare differently weighted quantities across cases.

### A8 — Add symmetry and invariance regression tests (P1)

Beyond recovering one synthetic gamma, verify that periodic spatial translation leaves powers/growth unchanged; changing field amplitude rescales power but not gamma; reordering particle rows preserves weighted moments; splitting/merging equal-state particle weights preserves estimators; and rotating fields, velocities and coordinates together preserves scalar diagnostics in supported geometries. Test geometry-specific routines only against transformations they claim to support. These tests target unit, axis and weighting errors that attractive figures and a single reference run can miss.

Recommended insertion into the execution sequence: resolve A1 with P0 measurement conventions; implement A2/A3/A5 before production regeneration; add A4/A7/A8 to estimator validation and A6 to reproducibility work. None of these additions requires changing production plasma parameters.

## 9. Implementation status after the requested code changes

The audit above is the historical baseline. This section supersedes its initial
"planning only" status. Existing user edits were preserved and extended; simulation
parameters and historical results/v5 products were not modified.

| Work | Status in this revision |
|---|---|
| Version/source/configuration/input provenance, strict numeric JSON | Implemented in `analysis_contract.py` and the manifest/diagnostic producers |
| Scientific evidence matrix and traceable HTML report | Implemented in `quality_report.py`; generated for all three local v5 cases |
| Energy time alignment and cumulative closure coverage | Implemented with explicit default-zero rounding tolerance; production explanation still needs original diagnostics |
| VDF coordinate labels and cylindrical normalization | Implemented: v displays, explicitly u-space initialization fit, exact annular density and retained probability |
| Solver objective and particle holdout | Implemented with fallback provenance, effective counts and conditional fit uncertainty; no automatic physical model identification |
| Early modal candidates and separate fastest accepted candidate | Implemented; strongest-power reference preserved, branch classification remains evidence-driven |
| Correlated growth-fit error | Implemented Bartlett HAC estimate, retaining fit/window components; synthetic regression coverage added |
| Structure centring, pressure smoothing and co-location | Implemented, including first/final threshold/smoothing sensitivity |
| Structure tracking | Implemented conservative periodic-label overlap with split/merge metadata; not an advection-aware tracker |
| Heat-flux averaging and distribution-matched noise estimate | Implemented integrated-ratio output and truncated empirical influence estimate, explicitly marked asymptotic |
| Deterministic task sampling | Implemented for integrated particle analysis, VDF display, heat flux and estimator comparison |
| Ordered execution and resumability | Implemented standalone runner with isolated stage output, exclusive writer lock, atomic records, fingerprints and logs |
| Comparison guards and invalid/gapped plots | Implemented; incompatible estimators are rejected and invalid growth measurements remain unavailable |
| Symmetry, normalization, candidate and metadata regression tests | Implemented in `test_analysis_revision.py` plus existing synthetic suite |

**Follow-up of 2026-09-29** (details in `IMPLEMENTATION_V6.md`): a v6 COSMA
job (`reanalysis_v6_all.sh`) now exists and was run end to end on synthetic
runs; the runner's stage collision, the gamma(k) zoom that hid the growing mode,
the legacy-estimator PASS and the sample-size-blind initial tolerances are
fixed; figure content checks, the energy audit and a spatial-mixture control of
the kappa tail are implemented; the 16 stale case comments are corrected. The
energy audit shows that the anomaly is **not** specific to kappa 5: all three
runs heat their electrons x8.3–8.6, isotropically and within 3.1 % of each
other, with correctly mapped DiagEnergies columns; all three FAIL the budget.

**Remaining work is explicit, not represented as completed:**

- Production regeneration requires submitting `reanalysis_v6_all.sh` on COSMA.
  The mechanism of the electron heating requires controlled reruns (ppc, dx,
  shape order; proposal in `IMPLEMENTATION_V6.md`); post-processing cannot
  repair the trajectories.
- Oblique kinetic theory, solver selection and external-reference validation are
  not implemented. The existing parallel solver still must not be used as a
  mirror prediction.
- A controlled grid/dt/ppc/box/seed campaign requires new simulations; no plasma
  parameters or jobs were changed. Existing convergence tooling is retained,
  with added estimator-version and duplicate-input guards.
- Full nonlinear confidence intervals and calibration of the matched heat-flux
  floor on production tails remain research validation. The spatial-mixture
  control bounds the tail a resolved temperature spread can produce; it does
  not see variation below the macro-cell size. Conditional covariance or a held-out score is
  not advertised as resolving those questions.
- Full cache/refactor work, production I/O benchmarking, advection-aware
  structure tracking and a final publication figure selection remain outside
  this code revision. Figure content is now checked at save time (series that
  draw nothing, all-NaN maps, empty panels); pixel-level comparison at thesis
  size is not implemented.
- The two monolithic reconnection cases still await migration to a thin case
  header (a structural change that needs a decision); they are the only
  remaining parity findings.

Verification: the full analysis suite passed with **150 tests and 14 subtests**
after the main changes. The final regression additions and runner smoke check
are recorded in `IMPLEMENTATION_V6.md`. Do not interpret synthetic tests as
validation of the old production trajectories.
