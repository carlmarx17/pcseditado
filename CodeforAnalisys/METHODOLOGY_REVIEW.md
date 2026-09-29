# Physical review of the analysis methodology

Date: 2026-09-29. Scope: every diagnostic of `CodeforAnalisys` as used for the
moderate mirror series (beta_i,par = 5, A_i = 2, beta_e = 1, mi/me = 200,
20 d_i box, 576^2 cells, 1000 ppc) and its distribution twins (bi-Maxwellian,
kappa 5, kappa 3). For each method: the physical question it answers, whether
it can answer it with these runs, what limits its validity, and how much a
paper should rest on it. Numbers quoted come from the v5 deliveries and the
energy audit (`IMPLEMENTATION_V6.md`); none of this replaces regenerating the
products with v6 and running the isotropic controls.

## 1. Four facts that condition every analysis

1. **Two instabilities, not one.** At beta_i,par = 5 the marginal anisotropies
   for gamma = 1e-3 Omega_ci are A = 1.22 for the ion-cyclotron (IC) branch and
   A = 1.23 for the mirror (Hellinger et al. 2006 fits). With A = 2 both are
   strongly unstable. The v5 dispersion products put the strongest magnetic
   power at k_par d_i = 0.31, k_perp = 0 with compressibility ~ 1e-12: a
   parallel, transverse mode, the IC signature, not the mirror one. Every
   "mirror" measurement must therefore be made on the oblique compressive
   component (dB_par, k_perp != 0), and the IC branch identified separately
   (transverse, parallel, left-hand polarised: the circular-polarisation
   psi_+- diagnostic, whose sign convention must be checked against a
   synthetic left-hand wave before it is quoted).
   A growth rate from the total fluctuation or from the strongest-power mode
   is a mixture of both branches.
2. **Numerical electron heating.** All three runs heat their electrons x8.3–8.6
   (isotropically, within 3.1 % of each other) while the total energy rises
   64.6 % (kappa 5 DiagEnergies). T_e/T_e0 is already 1.45 at t Omega_ci = 20.
   Electrons carry 10 % of the ion perpendicular pressure at t = 0 and about
   as much as the ions by the end. Anything involving electron pressure, the
   total-pressure balance, beta_e-dependent thresholds or late-time
   saturation is affected. The isotropic controls quantify the numerical part;
   they remove it only if it is additive.
3. **Scale sampling.** rho_i = 2.24 d_i and Delta k d_i = 2 pi / 20 = 0.314, so
   the box samples k rho_i = 0.70, 1.40, 2.11, ...: two or three modes inside
   the unstable band. gamma(k) is sampled, not resolved, and wavelength
   selection cannot be studied; a 40 d_i twin (as for the firehose series)
   would double the sampling.
4. **One realisation per case.** The v5 growth-rate differences between the
   distributions are 15–20 %, the fit errors ~ 3 %. Without a second seed per
   case the realisation-to-realisation spread is unknown, and no difference
   between distributions is established yet.

## 2. Method by method

Importance: **A** = a paper conclusion rests on it; **B** = supporting evidence
or a validity gate; **C** = diagnostic/secondary.

| Method (script) | Physical question | Effectiveness and validity now | Imp. | Needed before citing |
|---|---|---|---|---|
| Initial state (`validate_moments`, `check_initial_conditions`) | Is the run the plasma the profile declares? | Sound: exact moments, shot-noise-aware tolerances; all three v5 runs PASS. Window only (4 % of the domain); T defined as m Var(u) | B | Nothing; state that it is a window check |
| Energy audit + controls (`energy_audit`) | Is the energy budget trustworthy, and how large is the numerical heating? | Strong: column mapping verified at t = 0, window and global agree (2.3 %), common-mode across distributions. All runs FAIL. The control subtraction is exact only for additive heating | **A** (validity gate) | Run the three isotropic controls; ideally one ppc and one dx variant to identify the mechanism |
| Field residuals (`field_residuals`) | Does the solver keep div B, Gauss and continuity? | Sound: div B at round-off; Gauss/continuity from PSC logs, needs the job logs | B | Pass the COSMA job logs (`LOGS=`) |
| Modal growth rate (`growth_fit`, `physical_diagnostics`) | gamma of the instability | Sound numerics (transient-aware window, 2x gain, HAC error). Valid only per branch: the strongest-power mode can be the IC wave | **A** | Fit the oblique dB_par mode for the mirror and the parallel dB_perp mode for IC; state T_e drift in the window |
| gamma(k_par, k_perp) maps (`growth_rate_map`) | Which wavevectors grow, at what angle? | Best tool to separate branches (parallel vs perp component). Coarse: 2–3 modes in the band | **A** | Use the parallel-component map for the mirror; a 40 d_i twin for resolution |
| Mode identification (`dispersion_modes`, compressibility, helicity) | Is the growing mode mirror (aperiodic, compressive, oblique) or IC (propagating, transverse, parallel)? | Adequate; the frequency resolution is 2 pi / T_window (0.52 Omega_ci for a 12 Omega_ci^-1 window), so "omega ~ 0" means "below resolution" | **A** | Report angle, compressibility and polarisation per candidate, never the case label |
| omega–k, P+-(k, omega) (`dispersion_analysis`, `polarization_dispersion`) | Real frequency and polarisation of propagating modes | For the mirror (aperiodic) only a consistency check; for the IC competitor the handedness (psi_+-) is the identifying evidence | B (IC) / C (mirror) | Use it to confirm the IC branch, not to measure the mirror |
| Anisotropy evolution and Brazil plots (`anisotropy_analysis`) | Does the plasma relax to marginal stability? | Sound measurement. The drawn thresholds are bi-Maxwellian, cold-electron approximations: they are neither the kappa thresholds nor valid once beta_e has grown | **A** | Thresholds for the actual distribution and measured beta_e(t), from a kinetic solver |
| Structures and pressure balance (`structures_analysis`, `structure_tracking`) | Holes or peaks, their size, lifetime and pressure balance | Physically sound (mean-centred, co-located, sensitivity sweeps, periodic tracking). Pressure balance includes P_perp,e, which the numerical heating inflates up to ~ P_perp,i | **A** | Pressure balance with the control's electron heating subtracted, or ion-only balance shown separately |
| kappa_eff and VDFs in holes/peaks (`vdf_spatial`, `kappa_eff`) + Liouville closures (`liouville_kappa`) | Do the tails respond to the local field as mu-conserving particles would? | kappa_eff (whitened, truncated kurtosis) is the robust estimator; the new spatial-mixture control bounds tails a temperature spread can fake. The adiabatic mapping assumes structures >> rho_i; here structures are a few rho_i, so it is a limiting model | **A** (the distinctive result) | Mixture fraction per step; state the adiabatic assumption and test it against the trapped fractions |
| kappa fit of the window VDF (`physical_diagnostics`) | Global tail index | Fragile: 1-D marginal in u, bounded at 80 (Maxwellian-consistent), sensitive to sample size | C | Prefer kappa_eff; cite fits only with their conditional error |
| Heat flux (`heat_flux_analysis`) | Is there net suprathermal energy transport? | Correct third moment with truncation and a matched null floor; in a gyrotropic, statistically homogeneous box the expected mean flux is zero, so a signal must exceed the floor | C (B if above the floor) | Report only where above the matched floor |
| Field–particle exchange J·E (`energy_exchange`) | Which species gives energy to the fields? | Sound in principle; current from moments unless deposited current is saved, staggering and cadence limit the closure; blind to part of the grid heating | B | Closure against the controls; deposited current if available |
| Estimator consistency (`estimator_consistency`) | Do particle and moment anisotropies agree? | Good quality control; differences expose frame/window effects | B | None |
| Diamagnetic current (`diamagnetic_current`) | Magnetisation current of the structures | Derived quantity, gradient- and smoothing-sensitive; not the full current | C | Only as illustration |
| E(k,t), 3-D VDF scatter, magnetic spectrum | Overview | Descriptive; the spectrum index is now reported only over a real decay range | C | None |
| Comparison across distributions (`compare_physical_cases`, `kappa_evolution`) | Does the tail change the instability? | Guards against mixed estimators and gaps. Cannot yet separate a distribution effect from realisation noise | **A** | Second seed per case (or ensemble statistics of structures); branch-resolved gamma |
| Convergence (`convergence_study`) | Are the results numerical-resolution independent? | Tool ready; no runs | **A** (missing) | dx or ppc variant of one case at least |
| Linear theory (`linear_theory`) | Expected gamma(k) | Parallel propagation only; no oblique solver, no bi-kappa oblique, cold electrons in the Brazil thresholds | **A** (missing) | An oblique bi-kappa solver (e.g. LEOPARD) with hot, measured electrons |

## 3. What the data can support today, and what they cannot

Supported, with the caveats above: the initial states match the profiles;
the magnetic fluctuations grow and saturate; the anisotropy relaxes; the runs
develop compressive structures whose statistics and local VDFs can be measured
robustly; the tails respond to the local field and a spatial-mixture origin
can be bounded.

Not supported yet: a mirror growth rate distinct from the ion-cyclotron one;
a ranking of the distributions by growth rate or saturation level; any
energy-transfer or late-time pressure-balance statement that involves the
electrons; agreement with linear theory.

## 4. Priority order to reach a publishable set

1. Regenerate everything with v6 (`reanalysis_v6_all.sh`) and read the evidence
   report: modal gamma per branch, figure checks, energy audit.
2. Run the three isotropic controls; report baseline-corrected electron heating
   and closure.
3. Separate the branches: mirror gamma from the oblique dB_par modes, IC from
   the parallel dB_perp modes with their polarisation.
4. Kinetic thresholds and gamma(k) for bi-kappa with the measured beta_e.
5. One more seed per distribution (short runs to t Omega_ci ~ 60 suffice for the
   linear phase) and, if possible, a 40 d_i mirror twin.
