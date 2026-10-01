# Guide to the new analyses and figures (v6c)

What each new figure shows, why it is built that way, how to read it, and which
question of the thesis or the paper it answers. Paths are relative to
`analysis_results/v6c/`. Numbers quoted are from the v6b products (three
moderate mirror runs, 6 particle snapshots); v6c repeats them with ~120.

The images are the v6b figures, linked from `analysis_results/v6b/` next to this
folder. That tree is not tracked in git (data policy), so they show in a local
preview once the figures exist there (run the three commands of section 7, or
rsync the folders from COSMA) and not on GitHub. Sections 4 and 5d have no image
yet: their figures need the v6c job.

Run names: `<case>` is e.g. `mirror_bikappa3_moderate`; `<series>` is
`mirror_moderate_kappa`.

---

## 1. Which mode grows, and how fast

| Figure | Where |
|---|---|
| `mode_amplitude.png`, `growth_rate_vs_kappa.png` | `paper_figures/<series>/` |
| `growth_rate_fit_mode*.png` | `<case>/09_physical_diagnostics/` |
| `polarization_dispersion_{plus,minus}_*.png`, `mode_growth_*.png` | `<case>/04_spectra/` |

![mode amplitude](../analysis_results/v6b/paper_figures_new/mode_amplitude.png)

![growth rate vs kappa](../analysis_results/v6b/paper_figures_new/growth_rate_vs_kappa.png)

**Why.** The domain rms of dB is dominated by particle noise and hides the
growth; the amplitude of the dominant Fourier mode rises two decades. Curves
start after the quiet-start noise has settled (t > 2/(k v_th,i), stated on the
figure), because t = 0 and the noise build-up distort a log axis.

**How to read.** Straight segment on the log axis = exponential growth; the
dashed line is linear theory at the same k. `growth_rate_vs_kappa` puts the
three measured rates on the theory curve against 1/kappa.

**What it is for.** The central quantitative result: the mode is the parallel
ion-cyclotron wave (k_par d_i = 0.31, k_perp = 0, left-handed), and its growth
rate drops with the suprathermal tail, in agreement with kinetic theory within
2 %.

## 2. Why the tail lowers the growth rate

`paper_figures/<series>/resonance.png`

![resonance](../analysis_results/v6b/paper_figures_new/resonance.png)

Reduced f(v_par) of the three distributions at equal T_par with the
cyclotron-resonant velocity marked. A kappa plasma of the same temperature has
fewer ions at the resonance (0.89 for kappa 5, 0.78 for kappa 3, relative to
the Maxwellian): fewer resonant ions, slower growth. This is the physical
explanation of section 1.

## 3. Where the plasma sits relative to the thresholds

`paper_figures/<series>/brazil_trajectories.png`

![Brazil trajectories](../analysis_results/v6b/paper_figures_new/brazil_trajectories.png)

(beta_i_par, A_i) trajectories with the mirror threshold computed with the
*measured* electrons (they heat numerically from beta_e = 1 to ~8) and the
ion-cyclotron contour. Shows that the runs relax towards the ion-cyclotron
contour, and why no mirror growth is measured.

### 3b. Mirror against ion-cyclotron

`mirror_ic_competition/mirror_drive.png` (script `mirror_ic_competition.py`,
from products)

![mirror drive](../analysis_results/v6b/mirror_ic_competition/mirror_drive.png)

Ion anisotropy and the mirror drive Gamma = beta_perp (A - 1) - 1 (Hellinger
2007, with the measured electrons and with cold electrons) through the runs.

**Result.** Gamma = 8.6 at t = 0 and about 1 at the end: the mirror mode is
linearly unstable for the whole run and still does not grow (oblique dB
0.003-0.008 B0 against 0.20-0.25 for the ion-cyclotron wave). The threshold
does not explain its absence. Candidates, not separated by these runs: the
ion-cyclotron mode grows at least as fast and removes the anisotropy; the box
is 8.9 rho_i, so its first oblique mode sits at k_perp rho_i = 0.70; the 2D
geometry favours the ion-cyclotron wave (Shoji et al. 2009). The near-threshold
mirror formula gives ~0.12 Omega_ci but is outside its validity (Gamma << 1);
a quantitative comparison needs an oblique kinetic solver or a larger box.

## 4. Velocity distributions

| Figure | Where | Shows |
|---|---|---|
| `kappa_comparison_{parallel,perpendicular}.png` | `<case>/03_particles/` | f(v) at t = 0 and at the end with a Maxwellian, the fitted kappa and the loaded kappa_0, all at the variance measured at that time; lower panels PIC/Maxwellian |
| `goodness_of_fit.png` | `<case>/03_particles/` | F_PIC - F_model with the 95 % KS band |
| `distribution_change_{ions,electrons}.png` | `<case>/03_particles/` | log10 f(v,t)/f(v,0) in time |
| `vdf_2d_<species>_step_*.png`, `vdf_planes_*` | `<case>/09_physical_diagnostics/` | f(v_par, v_perp) per d^3v with the initial model at the same contour levels |
| `vdf_evolution.png` | `paper_figures/<series>/` | f(v_par, v_perp) of every run at t = 0, end of linear phase, end, with the resonant velocities |

**Why.** A model is comparable with data only at the same variance. The old
figure drew kappa_0 with the *initial* temperature on the *final* snapshot and
looked wrongly normalised; the formula was right. kappa is now a
maximum-likelihood estimate with a bootstrap interval.

**How to read.** The t = 0 column of `kappa_comparison` is a check of the
loader: the fit must return kappa_0. In the ratio panel a tail is a rise at
large |v|. In `vdf_evolution`, change concentrated along the dotted resonance
lines is resonant scattering.

**What it is for.** Validation of the initial condition (methods section), and
the picture of where in velocity space the anisotropy is removed.

## 5. The kappa index in time and the magnetic field

All in `kappa_dynamics_<series>/` (script `kappa_dynamics.py`, runs from
products in seconds).

### 5a. `kappa_field_evolution.png`

![kappa and field evolution](../analysis_results/v6b/kappa_dynamics_mirror_moderate_kappa/kappa_field_evolution.png)

Three panels on one time axis: 1/kappa of the ions in the local-field frame
(0 = Maxwellian, kappa on the right axis); the fraction of ions beyond 3 sigma
in v_par relative to a Gaussian; the fluctuation energy <|dB|^2>/B0^2 (solid)
and its compressive part (dotted).

**Result.** kappa 3 -> 4.8 and 5 -> 8.9 by t Omega_ci = 158; the tail content
of the kappa runs only decreases. In the bi-Maxwellian run the tail content
*drops* to half the Gaussian value right after saturation and recovers: the
wave flattens the distribution, it does not create a tail.

### 5b. `kappa_relaxation.png`

![kappa relaxation](../analysis_results/v6b/kappa_dynamics_mirror_moderate_kappa/kappa_relaxation.png)

Fit ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c int W dt, W = <|dB|^2>/B0^2, and
the windowed relaxation rate against W.

**How to read.** In (b) a rate that rises with W is wave-driven; the intercept
at small W is the background nu_0. Dotted lines in (a) are the nu_0 part alone.

**Result.** One c = 0.066 +- 0.005 fits both kappa_0: the erosion of the tail
is proportional to the fluctuation energy with the same coupling. nu_0 =
(1.5-2.2)e-3 Omega_ci acts from t = 0, before the waves grow; the isotropic
controls (same kappa, no waves) decide whether it is numerical. Do not state
its origin before they are analysed.

### 5b-bis. `kappa_energy_collapse.png`

![kappa against wave energy](../analysis_results/v6b/kappa_dynamics_mirror_moderate_kappa/kappa_energy_collapse.png)

The same result as 5b in the form to show: (a) the index against the
accumulated fluctuation energy F = int W dt; (b) the wave-driven part of the
change, after removing nu_0 t, against F. Both kappa runs fall on one line of
slope -c. Prefer this figure over 5b in a talk.

### 5c. `kappa_vs_local_field.png`

![kappa vs local field](../analysis_results/v6b/kappa_dynamics_mirror_moderate_kappa/kappa_vs_local_field.png)

1/kappa against the local b = |B|/B_ref at several times, the slope
d(1/kappa)/d ln b and the |B|-hole minus peak difference in time.

**Result.** Flat at every time (slope and difference consistent with zero).
An ion crosses the particle window in ~2.5 Omega_ci^-1 while 1/kappa e-folds in
~300: kappa is a property of the whole population, not of the local field.

### 5d. `kappa_shape_evolution.png` (needs v6c)

f(v_par) in units of its own sigma(t): the change since t = 0 as a time x
velocity map, and f/Gaussian at four instants, next to reference shapes.

**How to read.** Red wings at |u| > 3 = a tail forming. Red shoulders near the
dashed resonance lines with blue core and wings = flattening by resonant
diffusion. Timing is read against the dash-dot line (end of linear phase).

**What it is for.** The direct, model-free test of the saturation result in
5a; it can also refute it.

### 5e. `kappa_evolution_<series>/kappa_evolution.png`

![kappa evolution, local vs global frame](../analysis_results/v6b/kappa_evolution_new/kappa_evolution.png)

Local-field index (thick) over the old global-B0 kappa fit (faint). The bump of
the faint curve at saturation is the fit misreading the flattening, not a tail.

## 6. Numerical controls

| Figure | Where | For |
|---|---|---|
| energy audit tables and report | `quality_report/` | electron heating x8.3-8.6 is numerical (same in all distributions); subtracted with the isotropic controls |
| `field_residuals.png` | `<case>/09_physical_diagnostics/` | div B and solver constraints over the run |
| `estimator_consistency.png` | `<case>/09_physical_diagnostics/` | anisotropy from particles and from moments, local and global frame |
| `heat_flux_map_*.png` | `<case>/06_heat_flux/` | q_par per block, truncated estimator; hatched = consistent with zero |

These belong in the methods/limitations section: they bound what is physics
and what is the numerical setup.

---

## What can be claimed now, and what waits

| Claim | Status |
|---|---|
| Ion-cyclotron mode, growth rate vs kappa agrees with theory | v6b, solid |
| No measurable mirror growth | v6b, solid |
| Why the mirror mode does not grow although Gamma > 0 | open; needs an oblique solver or a larger / 3D box |
| kappa increases in time; independent of local \|B\| | v6b, 6 snapshots; firmer with v6c |
| Wave-driven erosion with one coupling c | v6b, 2 runs x 5 points; needs v6c |
| Flattening at saturation, not a tail | v6b tail fraction (global frame); direct test is 5d in v6c |
| Origin of the background rate nu_0 | open; needs the isotropic kappa controls |

## 7. Regenerating the linked figures (seconds, from products)

```bash
R=../analysis_results/v6b
python kappa_dynamics.py $R/mirror_bimaxwellian_moderate $R/mirror_bikappa5_moderate $R/mirror_bikappa3_moderate --outdir $R/kappa_dynamics_mirror_moderate_kappa
python paper_figures.py $R/mirror_bimaxwellian_moderate $R/mirror_bikappa5_moderate $R/mirror_bikappa3_moderate --outdir $R/paper_figures_new
python mirror_ic_competition.py $R/mirror_bimaxwellian_moderate $R/mirror_bikappa5_moderate $R/mirror_bikappa3_moderate --outdir $R/mirror_ic_competition
python kappa_evolution.py --case "bi-Maxwellian=$R/mirror_bimaxwellian_moderate/09_physical_diagnostics" --case "kappa5=$R/mirror_bikappa5_moderate/09_physical_diagnostics" --case "kappa3=$R/mirror_bikappa3_moderate/09_physical_diagnostics" --outdir $R/kappa_evolution_new
```

Detailed rationale and the defects each change fixed: `IMPLEMENTATION_V6.md`,
follow-ups 7 and 8.
