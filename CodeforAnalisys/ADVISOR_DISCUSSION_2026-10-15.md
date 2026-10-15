# Advisor discussion — mirror/EMIC with bi-kappa ions

Agenda and open questions from the v6-reviewed-20261015 analysis of the three
moderate mirror runs (bi-Maxwellian, bi-kappa 5, bi-kappa 3). Everything here is
measured from that tree unless marked as literature or as an assumption.

Written in English per the project convention so collaborators outside the group
can follow it.

## 1. What the runs are

Three runs that differ **only** in the ion velocity distribution:
beta_i_par = 5, A_i = T_perp/T_par = 2, beta_e_par = 1, A_e = 1, m_i/m_e = 200,
box 20 d_i, 576^2, 1000 ppc, to t = 158 / Omega_ci. The bi-kappa is loaded at
fixed second moment (same temperature as the bi-Maxwellian), a multivariate
Student-t with nu = 2 kappa - 1 whose scale is shared by the three velocity
components — a bi-kappa, not a product bi-kappa.

The measured index at t = 0 is 1/kappa = 0.1988 and 0.3326 against the nominal
0.2 and 0.3333, so the loader reproduces the intended distribution to 0.6 % and
0.2 %.

## 2. Results to present, strongest first

### 2.1 The core relaxes identically, the tail does not

Final ion anisotropy by parallel-velocity band:

| band v_par/v_A | bi-Maxwellian | kappa 5 | kappa 3 |
|---|---|---|---|
| 0 – 1.6 | 1.247 | 1.278 | 1.244 |
| 1.6 – 3.2 | 1.278 | 1.330 | 1.272 |
| 3.2 – 4.7 | 1.299 | 1.359 | 1.363 |
| **> 4.7** | **1.368 ± 0.014** | **1.741 ± 0.020** | **1.839 ± 0.025** |

The cores relax to A ~ 1.24–1.33 in every run. All of the difference between the
runs sits in the suprathermal band. kappa 3 minus bi-Maxwellian in that band is
0.471 ± 0.029, 16 sigma.

Internal control against the "finite macro-particles" objection: the
bi-Maxwellian run's own suprathermal band ends at 1.368 while its marginal
anisotropy is 1.367. It sits exactly at marginal stability, so a residual
unrelaxed tail is not something the sampling produces by itself.

### 2.2 The saturation state scales with 1/kappa

| | t_sat | gamma_linear at sat | A_i - A_marginal | fluctuation energy |
|---|---|---|---|---|
| bi-Maxwellian | 126.2 | 0.0015 | 0.0066 | 0.0687 |
| kappa 5 | 139.8 | 0.0087 | 0.0391 | 0.0504 |
| kappa 3 | 139.4 | 0.0174 | 0.0813 | 0.0446 |

Suprathermal ions make the instability saturate later, with ~35 % less wave
energy, and leave the plasma an order of magnitude further above marginal
stability — still linearly unstable.

### 2.3 The normalisation convention decides the sign, and the PIC discriminates

Linear theory at the measured mode k_par d_i = 0.314, same parameters:

| kappa | same temperature (ours) | same core (Lazar 2015) | PIC measured |
|---|---|---|---|
| inf | 0.1236 | 0.1236 | 0.1231 |
| 5 | 0.1194 | 0.1413 (beta 7.14) | 0.1220 |
| 3 | 0.1134 | 0.1544 (beta 10.0) | 0.1120 |

The same-temperature prediction matches the simulation to 0.4–2.2 %; the
same-core one is 38 % off at kappa = 3. The published statement that
suprathermals *enhance* anisotropy instabilities follows from the same-core
normalisation raising beta (5 -> 10 at kappa = 3), not from the suprathermal
population itself.

This also validates the whole chain end to end: loader convention, dispersion
solver and growth fit agree with the simulation at the 1 % level.

### 2.4 The quasilinear closure fails

Quasilinear treatments of bi-kappa proton instabilities assume the distribution
"maintains the bi-Kappa form for all time". Measured here, the index moves from
3 to ~4.8 and from 5 to ~9 (local-field frame) over the run. Lazar et al. say
this is exactly what quasilinear theory cannot capture; our runs quantify it for
ions.

The relaxation fits ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c * int W dt with
r^2 = 0.998. Neither term alone works (chi2/dof 54–98 for the background alone,
218–255 for the wave term alone, 2.3–3.1 together). The wave coefficient is
nearly the same in both runs (c = 0.0676 and 0.0646); the background rate is not
(nu_0 = 0.00210 and 0.00149).

### 2.5 Negative result worth reporting as such

The index measured inside magnetic holes versus peaks differs by less than half
a percent, and the difference is *largest* in the bi-Maxwellian run (-7.9 sigma)
which has no suprathermal population. The sign agrees with the thermodynamic
expectation (kappa lower where |B| is higher), but the bi-Maxwellian control
argues that it is how the distribution is sampled between holes and peaks, not a
property of the index. Report as bounded null, not as a finding.

Also: over the whole box, <|B|>/B0 - 1 = 0.96 * <|dB|^2>/2B0^2 at every time in
all three runs, so the box-averaged field magnitude carries no information the
fluctuation amplitude does not already carry. "Magnitude, not fluctuation" is
only a distinct measurement locally.

## 3. Theoretical gaps — the main agenda

### T1. The resonance ceiling is set by the box (most serious)

The n = 1 cyclotron resonance of the dominant mode sits at |v_res|/v_A = 1.84,
1.87, 1.93 for the three runs. The bands relax monotonically with distance from
it (1.247, 1.278, 1.299, 1.368 for the bi-Maxwellian), which supports the
resonance-exclusion mechanism. The compressive mirror branch has omega_r ~ 0 and
resonates near v_par ~ 0 through transit-time damping, which covers the core;
between the two the plasma is scattered from 0 to ~2 v_A.

**But** v_res = (1 - omega/Omega_ci) / (k_par d_i), and the smallest k the box
admits is k_min d_i = 2 pi / 20 = 0.314. The box cannot provide a mode that
resonates above ~1.9 v_A. So the unrelaxed tail above ~2 v_A may be a box-size
artifact rather than physics.

Question for the advisor: how do we separate these? The 40 d_i firehose runs
have k_min d_i = 0.157 and therefore roughly twice the resonance ceiling, so
they are a natural test — if the tail there is unrelaxed above ~4 v_A rather
than above ~2 v_A, the effect follows the box and not the physics.

### T2. There is no bi-kappa mirror threshold in the pipeline

`mirror_threshold_electrons` is bi-Maxwellian theory; its own docstring says
that for bi-kappa ions it is "a reference, not the kappa threshold". So the
excess anisotropy of section 2.2 — one of the headline numbers — is measured
against a bi-Maxwellian marginal line for all three runs.

Leubner & Schupfer (2002) and Lopez et al. (2023) have bi-kappa mirror
thresholds. Do we implement one, or do we report the excess explicitly as
"above the bi-Maxwellian marginal line" and argue that the comparison is still
differential because the same line is used for all three runs?

Note that Lopez et al. had to rescale their analytical bi-kappa formulas by
about 1.4 (mirror) and 5.9 (cyclotron) to match an exact solver, so the
analytical thresholds are not quantitative.

### T3. No oblique kappa dispersion solver

`ParallelDispersion` solves parallel propagation only. The mirror mode measured
here is oblique (theta = 63–68 degrees). We therefore have a theory curve for
the transverse ion-cyclotron branch and none for the compressive branch, which
is why section 2.3 uses the transverse mode. Do we need an oblique bi-kappa
solver (DIS-K exists, Lopez et al. 2021) to say anything quantitative about the
mirror branch itself?

### T4. The background relaxation rate nu_0 has no theory

It acts from t = 0, before there are waves, and does not follow beta. Candidate
explanations: collisionless phase mixing, numerical collisionality from particle
noise, or a genuine relaxation channel. Yoon's self-consistent framework
predicts the opposite sign (turbulence builds the tail). The isotropic controls
were built to measure exactly this and are not yet available.

### T5. Why is the wave coefficient c independent of kappa_0?

c = 0.0676 (kappa 5) and 0.0646 (kappa 3), a 5 % difference, while nu_0 differs
by 40 %. If the erosion were resonant pitch-angle diffusion one might expect it
to depend on how many particles sit in resonance, which depends on kappa. No
explanation at present.

### T6. Two kappa estimators disagree by 40 %

The global-B0 fit (`kappa_evolution.py`) gives delta(1/kappa) = -0.183 for
kappa 3; the local-field maximum-likelihood estimate (`kappa_dynamics.py`) gives
-0.125. The documented reason is that the global fit reads the wave-tilted
distribution at saturation as a tail. Which one do we quote when comparing with
theory, and how do we justify it?

### T7. bi-kappa versus product bi-kappa

Our loader shares one scale factor across the three velocity components, giving
a true bi-kappa (multivariate Student-t). Much of the literature uses a product
bi-kappa, whose dispersion relation differs. We need to establish which form
each comparison paper uses before claiming agreement or disagreement — this is a
second convention trap on top of the temperature/core one.

### T8. The electrons develop a small perpendicular deficit

A_e drifts from 1.002 to ~0.975 in every run while beta_e grows by a factor 8.4.
The drift is common-mode, so it cancels in the comparison, but it is
unexplained. Is it physical (preferential parallel heating) or numerical?

## 4. Numerical and validation gaps

- **All three runs carry scientific_status = FAIL** from the energy budget:
  electrons gain 6.3x what the ions and the magnetic fluctuations release
  (T_e x8.3–8.6), and in the kappa 5 run the total energy changes by +64.6 %.
  The defence available today is common mode: the heating differs by 3.7 %
  between three very different distributions, so it cannot produce a factor 12
  difference in excess anisotropy. The isotropic controls would make this
  quantitative; they are incomplete (78 % and 51 %) and one failed on
  configuration.
- **Convergence is UNVERIFIED and the resolution runs were never submitted.**
  `res_ppc4000` and `res_ngrid1152` exist as job scripts but are not on disk.
  This matters here more than usual because the result lives in the
  low-population tail. Budget: about 5400 node-hours for both, ~1800 for
  ppc4000 alone.
- **2D only.** Restricting wavevectors to a plane removes oblique pitch-angle
  scattering channels, which is the strongest anticipated objection to a claim
  about an unrelaxed tail.
- **No usable mirror growth rate.** The compressive fits are 2.83 ± 2.52
  (kappa 5) or do not converge. They need a hand-set window before any mirror
  gamma can be quoted.
- **The measured mode is the box fundamental**, k_par d_i = 0.314, while linear
  theory peaks at ~0.36. We are measuring the nearest allowed mode, not the
  maximum-growth mode, and should say so.

## 5. Questions to put to the advisor

1. T1 is the one that can invalidate the central result. Do we run the 40 d_i
   firehose pair as the box test before anything else? The data exist (88 % and
   77 % complete) and the analysis costs about 20–30 node-hours.
2. Do we spend ~1800 node-hours on the ppc4000 convergence run, or do we argue
   from the bi-Maxwellian internal control in 2.1?
3. Do we implement a bi-kappa mirror threshold (T2) or report the excess
   anisotropy against the bi-Maxwellian line with that stated?
4. Which kappa estimator is the one we quote (T6)?
5. Is the paper framed as "first PIC with bi-kappa ions" or as "the reported
   sign of the suprathermal effect is a normalisation artifact, and here is the
   simulation that decides"? The second is stronger and less confrontational,
   and does not depend on the novelty claim surviving the literature search.
6. The electron heating: do we report the runs as they are with the common-mode
   argument, or do we hold the paper until the controls are complete?

## 6. What can be done with analysis only, no new simulation

- The firehose pair (88 % and 77 %), with `ALLOW_INCOMPLETE=1` — a second
  instability family with the opposite sign of anisotropy, and the box test of
  T1.
- The incomplete isotropic controls, which give a baseline over the time they
  cover (to t ~ 123 and ~ 81, i.e. through the linear phase).
- Re-fitting the compressive growth rates with a hand-set window.
- `LOGS=` and an `analysis_config.json`, which turn the `gauss`, `continuity`
  and `runtime` checks from UNVERIFIED into verdicts.

Everything else — convergence, a complete energy baseline, 3D — needs compute.
