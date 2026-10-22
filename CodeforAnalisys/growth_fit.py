#!/usr/bin/env python3
"""
growth_fit.py — one linear-phase growth-rate fit for the whole pipeline
=======================================================================
Before this module three scripts fitted gamma three different ways:
``physical_diagnostics.py`` selected the linear phase by amplitude,
``spectral_analysis.py`` (and through it ``polarization_dispersion.py``) by
the sign of the local slope -- which keeps saturation samples whenever noise
makes the slope flicker positive and biases gamma low -- and none of them
reported how much gamma moved when the window moved. A thesis table that
quotes gamma from two of those scripts would then compare two estimators.

The fit implemented here, for an amplitude a(t) ~ exp(gamma t):

1. ``y = ln a``, smoothed with a running median (width ~ n/25) only to
   *locate* the phases; the regression uses the raw samples.
2. Saturation is the maximum of the smoothed curve; the noise floor is the
   minimum of the smoothed curve *before* that maximum. Using the minimum
   instead of "the first 5 % of the series" matters for the whistler cases:
   they grow in a few Omega_ce^-1, i.e. within the first handful of
   snapshots, and a fixed head fraction would already contain the growth.
   It also skips the quiet-start relaxation that some runs show at t ~ 0.
3. The rise is the first contiguous stretch from the floor between
   ``floor + lo*(sat-floor)`` and ``floor + hi*(sat-floor)`` (default
   10 %-90 % of the rise in log amplitude).
4. Inside the rise, the linear phase is a contiguous interval around a
   maximum of the local log-slope where that slope stays >= ``slope_frac``
   (default 80 %) of that maximum. The steepest such interval is compared
   with the steepest one of what follows it, repeatedly, and the one with the
   largest gain in log amplitude is kept: the PIC quiet-start build-up of the
   noise floor is steeper than any instability but lasts ~2 Omega_ci^-1, and
   picking the steepest interval alone returned gamma ~ 0.6 on t < 2.5 for
   the v5 mirror runs (test_growth_fit.py reproduces it). Without the slope
   criterion the band includes the
   saturation roll-over: for a quasi-linear (logistic-like) saturation the
   local rate at 90 % of the log-rise is already ~0.4 gamma and the band fit
   comes out ~15 % low (measured on synthetic_run.py). With the criterion
   the same series gives gamma within ~5 % (1-2 gamma_err) from 15 to 2400
   snapshots and 0-20 % multiplicative noise (test_growth_fit.py). The slope
   is a local least-squares slope, applied contiguously and never outside
   the rise, so noise spikes in the saturated phase cannot enter the window.
5. Ordinary least squares on that window gives gamma and its standard
   error; refitting with other band edges and slope fractions gives the
   window sensitivity. ``gamma_err`` combines both in quadrature -- this is
   the number that belongs in an error bar.

6. ``fit_logistic_saturation`` fits the saturating curve itself -- no window,
   no threshold -- and its ``gamma_logistic``, ``t_saturation`` and
   ``amplitude_saturation`` are reported beside the windowed fit. The two
   gammas agree to 1 % on the v6 mirror runs; a run where they do not is a
   run whose linear phase the window search got wrong.

An explicit window (``t_start``/``t_end``, in the same units as ``time``)
bypasses steps 2-3; the sensitivity then comes from dropping one sample at
either end. Use it once the linear phase has been identified by eye or from
the growth-rate map.

For a power P ~ a^2 the growth rate is gamma = (1/2) d ln P / dt; callers that
fit a power must pass sqrt(P).
"""

from __future__ import annotations

import numpy as np

#: Alternative (lo, hi, slope_frac) choices used to measure the sensitivity of
#: gamma to the automatic window.
SENSITIVITY_WINDOWS = ((0.05, 0.85, 0.8), (0.15, 0.95, 0.8), (0.10, 0.90, 0.7),
                       (0.10, 0.90, 0.9))


def _running_median(y: np.ndarray) -> np.ndarray:
    """Running median of odd width ~ n/25 (3 to 51 samples).

    Used only to *locate* the floor, the rise and the saturation. A median
    leaves a monotonic ramp unbiased (the median of monotonic samples is the
    central one), so a width that grows with the number of snapshots
    suppresses the noise of dense series without shifting the linear phase.
    """
    width = int(np.clip(y.size // 25, 3, 51)) | 1
    if y.size < width:
        return y.copy()
    half = width // 2
    padded = np.pad(y, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _ols(t: np.ndarray, y: np.ndarray) -> dict:
    n = t.size
    t_mean = float(np.mean(t))
    sxx = float(np.sum((t - t_mean) ** 2))
    if n < 2 or sxx <= 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "stderr": float("nan"), "r_squared": float("nan")}
    slope = float(np.sum((t - t_mean) * (y - np.mean(y))) / sxx)
    intercept = float(np.mean(y) - slope * t_mean)
    residual = y - (slope * t + intercept)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if n > 2 else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "stderr": stderr,
            "r_squared": float(r2)}


def correlated_slope_error(t, y, fit):
    """Bartlett HAC slope error; lag is an explicit finite-sample rule, not seed spread."""
    t, y = np.asarray(t), np.asarray(y)
    n = len(t)
    if n < 4:
        return float('nan'), 0
    x = t - t.mean()
    sxx = np.dot(x, x)
    if sxx <= 0:
        return float('nan'), 0
    residual = y - (fit['slope'] * t + fit['intercept'])
    score = x * residual
    lag = min(n - 2, max(1, int(4 * (n / 100) ** (2 / 9))))
    meat = np.dot(score, score)
    for k in range(1, lag + 1):
        meat += 2 * (1 - k / (lag + 1)) * np.dot(score[k:], score[:-k])
    return float(np.sqrt(max(meat, 0) * n / (n - 2)) / sxx), lag


def _local_slope(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d y/d t by least squares over a centred window of ~1/5 of the rise.

    Finite differences of a noisy log-amplitude are dominated by the noise
    when the snapshots are dense; a local regression over a window that
    scales with the length of the rise keeps the slope estimate stable
    for both the whistler (few samples) and the mirror (hundreds) series.
    """
    n = t.size
    half = max(1, n // 10)
    slope = np.empty(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        tt, yy = t[a:b], y[a:b]
        dt = tt - tt.mean()
        den = float(np.sum(dt * dt))
        slope[i] = float(np.sum(dt * (yy - yy.mean())) / den) if den > 0 else 0.0
    return slope


def _auto_window(ys: np.ndarray, t: np.ndarray, lo: float, hi: float,
                 slope_frac: float | None = 0.8) -> tuple[int, int] | None:
    """Index range [start, end] of the linear phase of the first rise, or None."""
    i_peak = int(np.argmax(ys))
    i_floor = int(np.argmin(ys[: i_peak + 1]))
    floor, sat = float(ys[i_floor]), float(ys[i_peak])
    if not sat > floor:
        return None
    lo_level = floor + lo * (sat - floor)
    hi_level = floor + hi * (sat - floor)
    start = None
    for i in range(i_floor, i_peak + 1):
        if ys[i] >= lo_level:
            start = i
            break
    if start is None:
        return None
    end = i_peak
    for i in range(start, i_peak + 1):
        if ys[i] > hi_level:
            end = i - 1
            break
    if end < start:
        return None
    if slope_frac is None or end - start + 1 < 4:
        return start, end
    seg_t, seg_y = t[start:end + 1], ys[start:end + 1]
    slope = _local_slope(seg_t, seg_y)
    best = None
    lo_idx = 0
    # The region around the steepest local slope, then the steepest region of
    # what follows it, and so on; the one with the largest gain in log
    # amplitude wins. A PIC quiet-start transient (the noise floor building up
    # within ~1 Omega_ci^-1) has the steepest slope of the whole series but
    # lasts a few snapshots; the exponential phase after the noise plateau is
    # slower but spans several e-folds. Taking the steepest region alone put
    # the mirror-run fits at t < 7 Omega_ci^-1 with R^2 < 0.5. Only later
    # regions are tried, so the saturation flank of a single rise -- whose
    # slope only falls -- never displaces the linear phase.
    while lo_idx < slope.size:
        j = lo_idx + int(np.argmax(slope[lo_idx:]))
        if not slope[j] > 0:
            break
        keep = slope >= slope_frac * slope[j]
        a = b = j
        while a - 1 >= lo_idx and keep[a - 1]:
            a -= 1
        while b + 1 < keep.size and keep[b + 1]:
            b += 1
        if b - a + 1 >= 4:
            gain = float(seg_y[b] - seg_y[a])
            if best is None or gain > best[2]:
                best = (a, b, gain)
        lo_idx = b + 1
    if best is None:
        # The slope is too noisy for the criterion to isolate a phase (a lone
        # spike): keep the whole rise, whose bias the sensitivity refits show.
        return start, end
    return start + best[0], start + best[1]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """1/(1+e^-z), evaluated so neither tail overflows."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softplus(z: np.ndarray) -> np.ndarray:
    """ln(1+e^z) without overflow: for large z it is z + ln(1+e^-z)."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


#: Fraction of gamma that marks the ends of the linear phase on the fitted
#: logistic. 0.9 reproduces the window the band/slope search returns on the v6
#: mirror runs (t = 49.0 / 44.7 / 46.7 against 53.2 / 45.3 / 50.8), so the
#: parametric and the heuristic window mean the same thing.
LINEAR_PHASE_RATE_FRACTION = 0.9


def fit_logistic_saturation(time, amplitude, max_iter: int = 300) -> dict:
    """Fit a noise floor plus a saturating exponential to the whole series.

    The model is the one the series actually has -- a constant noise floor in
    power under a logistic rise::

        a(t)^2 = floor^2 + [a_sat / (1 + exp(-gamma (t - t_sat)))]^2

    so nothing has to be windowed away. The band/slope search of
    ``_auto_window`` answers "where does the series stop being exponential?"
    with three thresholds (``band``, ``slope_frac`` and the running-median
    width); this fits the curve itself, and gamma, the saturation time and
    the saturation amplitude come out as parameters with a covariance. On the
    v6 mirror runs the two gammas agree to ~1 %, which is why
    ``fit_exponential_growth`` reports this one *beside* the windowed fit
    rather than replacing it: the quoted gamma does not move, and a run where
    the two disagree is a run whose linear phase the window search got wrong.

    The floor term covers the quiet noise plateau, but not the quiet-start
    build-up of the first ~2 Omega_ci^-1 (the spectrum filling from an
    unperturbed start), which is neither floor nor growth. Callers cut it with
    ``psc_units.noise_settling_time(k_di)`` before fitting, as
    ``physical_diagnostics.fit_mode_growth`` does.

    ``t_saturation`` is the inflection, where the local rate has fallen to
    gamma/2 and the amplitude is at a_sat/2. It is the one point of a logistic
    that does not depend on a threshold, and it is *not* the point where the
    run is saturated: on the v6 mirror runs the amplitude still reaches only
    half its maximum there. ``logistic_linear_phase_end`` is the end of the
    linear phase proper, where the rate is ``LINEAR_PHASE_RATE_FRACTION`` of
    gamma.

    The parameter errors are Bartlett HAC (the lag rule of
    ``correlated_slope_error``): the residuals of a logistic fitted to a PIC
    mode are smooth, so the plain covariance understates them several-fold.

    ``rms_ln_amplitude`` is the residual of the model, in log amplitude. A run
    whose mode grows a second time after the first roll-over is not a logistic
    and says so here: the v6 kappa = 5 run (a rebound at t ~ 75-85) gives
    0.16 against 0.06 for the other two.

    Returns ``logistic_fit_ok = 0`` with a reason whenever the fit does not
    reach a positive gamma with the inflection inside the sampled interval.
    """
    time = np.asarray(time, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    valid = np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0)
    empty = {
        "gamma_logistic": float("nan"), "gamma_logistic_stderr": float("nan"),
        "t_saturation": float("nan"), "t_saturation_stderr": float("nan"),
        "amplitude_saturation": float("nan"), "amplitude_noise_floor": float("nan"),
        "logistic_linear_phase_end": float("nan"),
        "rms_ln_amplitude": float("nan"), "logistic_n_points": 0,
        "logistic_fit_ok": 0,
        "logistic_reject_reason": "fewer than 8 positive finite samples",
    }
    if np.count_nonzero(valid) < 8:
        return empty
    order = np.argsort(time[valid])
    t = time[valid][order]
    y = np.log(amplitude[valid][order])

    smoothed = _running_median(y)
    i_peak = int(np.argmax(smoothed))
    ln_sat = float(smoothed[i_peak])
    ln_floor = float(np.min(smoothed))
    if not ln_sat > ln_floor:
        return {**empty, "logistic_reject_reason": "no rise above the noise floor"}
    mid = 0.5 * (ln_sat + ln_floor)
    t_mid = float(t[int(np.argmin(np.abs(smoothed[: i_peak + 1] - mid)))])
    span = float(t[i_peak] - t[0])
    p = np.array([ln_sat, 4.0 * (ln_sat - ln_floor) / span if span > 0 else 0.1,
                  t_mid, ln_floor], dtype=float)

    def model(q):
        """ln a and the Jacobian of the four parameters, in a stable form."""
        sig = _sigmoid(q[1] * (t - q[2]))
        ln_s = q[0] - _softplus(-q[1] * (t - q[2]))
        # weight = s^2/(floor^2+s^2), from a difference of logs so neither
        # exponential can overflow when the two scales are far apart.
        weight = _sigmoid(2.0 * (ln_s - q[3]))
        predicted = 0.5 * (2.0 * np.maximum(ln_s, q[3])
                           + np.log1p(np.exp(-2.0 * np.abs(ln_s - q[3]))))
        jacobian = np.column_stack([
            weight,
            weight * (1.0 - sig) * (t - q[2]),
            -weight * (1.0 - sig) * q[1],
            1.0 - weight,
        ])
        return predicted, jacobian

    predicted, jacobian = model(p)
    residual = predicted - y
    cost = float(np.dot(residual, residual))
    lam = 1e-3
    for _ in range(max_iter):
        jtj = jacobian.T @ jacobian
        try:
            step = np.linalg.solve(jtj + lam * np.diag(np.maximum(np.diag(jtj), 1e-12)),
                                   jacobian.T @ residual)
        except np.linalg.LinAlgError:
            return {**empty, "logistic_reject_reason": "singular normal equations"}
        trial = p - step
        try:
            new_predicted, new_jacobian = model(trial)
        except FloatingPointError:
            new_cost = np.inf
        else:
            new_residual = new_predicted - y
            new_cost = float(np.dot(new_residual, new_residual))
        if np.isfinite(new_cost) and new_cost < cost:
            relative = np.max(np.abs(step) / np.maximum(np.abs(p), 1e-12))
            p, predicted, jacobian = trial, new_predicted, new_jacobian
            residual, cost, lam = new_residual, new_cost, max(lam * 0.3, 1e-12)
            if relative < 1e-10:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break

    rms = float(np.sqrt(cost / t.size))
    # A series with no visible plateau drives the floor to zero, and its
    # Jacobian column with it: inverting the whole normal matrix would then
    # return NaN for *every* parameter, gamma included. Only the parameters
    # the data actually constrain are inverted; the others keep a NaN error.
    scale = np.linalg.norm(jacobian, axis=0)
    keep = np.flatnonzero(scale > 1e-8 * max(float(scale.max()), 1e-300))
    errors = np.full(4, np.nan)
    if keep.size:
        # Bartlett HAC sandwich, the same lag rule as correlated_slope_error.
        # The residuals of a logistic fitted to a PIC mode are smooth, not
        # independent: the plain covariance makes gamma look ~8x better
        # determined than it is (+-0.0003 against +-0.0024 on the v6 mirror
        # runs). The quoted error has to be the correlated one.
        block = jacobian[:, keep]
        score = block * residual[:, None]
        n = t.size
        lag = min(n - 2, max(1, int(4 * (n / 100) ** (2 / 9))))
        meat = score.T @ score
        for k in range(1, lag + 1):
            cross = score[k:].T @ score[:-k]
            meat += (1 - k / (lag + 1)) * (cross + cross.T)
        try:
            bread = np.linalg.inv(block.T @ block)
            covariance = bread @ meat @ bread * n / max(n - keep.size, 1)
            errors[keep] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        except np.linalg.LinAlgError:
            pass

    ln_sat, gamma, t_sat, ln_floor = (float(v) for v in p)
    reasons = []
    if not gamma > 0:
        reasons.append(f"gamma={gamma:.4g} is not growth")
    if not t[0] <= t_sat <= t[-1]:
        reasons.append(f"the inflection t={t_sat:.4g} is outside the sampled interval")
    # d ln a/dt = gamma/(1 + exp(gamma (t - t_sat))) once above the floor, so
    # the rate is a fraction f of gamma at t_sat + ln(1/f - 1)/gamma.
    f = LINEAR_PHASE_RATE_FRACTION
    offset = np.log(1.0 / f - 1.0) / gamma if gamma > 0 else float("nan")
    return {
        "gamma_logistic": gamma,
        "gamma_logistic_stderr": float(errors[1]),
        "t_saturation": t_sat,
        "t_saturation_stderr": float(errors[2]),
        "amplitude_saturation": float(np.exp(ln_sat)) if ln_sat < 300 else float("inf"),
        "amplitude_noise_floor": float(np.exp(ln_floor)) if ln_floor < 300 else float("inf"),
        "logistic_linear_phase_end": float(t_sat + offset),
        "rms_ln_amplitude": rms,
        "logistic_n_points": int(t.size),
        "logistic_fit_ok": int(not reasons),
        "logistic_reject_reason": "; ".join(reasons),
    }


def fit_exponential_growth(
    time,
    amplitude,
    t_start: float | None = None,
    t_end: float | None = None,
    band: tuple[float, float] = (0.10, 0.90),
    slope_frac: float | None = 0.8,
    min_points: int = 3,
    min_gain: float = 2.0,
    min_r2: float = 0.7,
    max_onset: float | None = None,
) -> dict:
    """Fit a(t) ~ exp(gamma t) on its linear phase. See the module docstring.

    Returns a dict that always contains ``gamma`` (NaN when nothing can be
    fitted), ``fit_ok`` and ``fit_reject_reason``. ``fit_ok`` is 1 only for a
    positive gamma, R^2 >= ``min_r2`` and an amplitude gain >= ``min_gain``
    inside the window: a clean *decay* also has a high R^2.
    """
    time = np.asarray(time, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    valid = np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0)
    empty = {
        **{k: v for k, v in fit_logistic_saturation([], []).items()},
        "gamma": float("nan"), "gamma_stderr": float("nan"), "gamma_hac_stderr": float("nan"), "gamma_hac_lags": 0,
        "gamma_window_spread": float("nan"), "gamma_err": float("nan"),
        "intercept": float("nan"), "r_squared": float("nan"),
        "rvalue": float("nan"), "n_points": int(np.count_nonzero(valid)),
        "fit_time_range": None, "linear_phase_start": float("nan"),
        "linear_phase_end": float("nan"), "amplitude_gain": float("nan"),
        "series_start": float(time[valid][0]) if np.any(valid) else float("nan"),
        "window_source": "none", "fit_ok": 0,
        "fit_reject_reason": "fewer than 4 positive finite samples",
        "time": time[valid], "ln_amplitude": np.log(amplitude[valid]),
        "fit_time": np.array([]), "fit_ln_amplitude": np.array([]),
        "fit_amplitude": np.array([]),
    }
    if np.count_nonzero(valid) < 4:
        return empty
    order = np.argsort(time[valid])
    t = time[valid][order]
    y = np.log(amplitude[valid][order])

    reasons: list[str] = []
    if t_start is not None or t_end is not None:
        t0 = -np.inf if t_start is None else float(t_start)
        t1 = np.inf if t_end is None else float(t_end)
        idx = np.flatnonzero((t >= t0) & (t <= t1))
        window = (int(idx[0]), int(idx[-1])) if idx.size else None
        source = "explicit"
        alternatives = []
        if window is not None:
            s, e = window
            alternatives = [(s + 1, e), (s, e - 1)]
    else:
        ys = _running_median(y)
        window = _auto_window(ys, t, *band, slope_frac=slope_frac)
        source = "auto"
        alternatives = [w for w in (_auto_window(ys, t, a, b, slope_frac=f)
                                    for a, b, f in SENSITIVITY_WINDOWS) if w is not None]
        if window is None:
            reasons.append("no rise above the noise floor")

    if window is None or window[1] - window[0] + 1 < min_points:
        # Nothing that looks like a linear phase: still report the slope of the
        # whole series so a decaying mode keeps its negative gamma, but never
        # flag it as a valid growth rate.
        if window is not None:
            reasons.append(f"linear window has fewer than {min_points} samples")
        if window is None or source == "auto":
            window = (0, t.size - 1)
            reasons.append("slope of the whole series reported instead")
        source = f"{source}-fallback"

    s, e = window
    tf, yf = t[s:e + 1], y[s:e + 1]
    main = _ols(tf, yf)
    spreads = []
    for alt in alternatives:
        a0, a1 = alt
        if a1 - a0 + 1 >= min_points:
            spreads.append(_ols(t[a0:a1 + 1], y[a0:a1 + 1])["slope"])
    spreads = [g for g in spreads if np.isfinite(g)]
    spread = float(np.max(np.abs(np.asarray(spreads) - main["slope"]))) if spreads else float("nan")
    stderr = main["stderr"]
    hac, hac_lags = correlated_slope_error(tf, yf, main)
    parts = [v for v in (max(stderr, hac), spread) if np.isfinite(v)]
    gamma_err = float(np.sqrt(sum(v * v for v in parts))) if parts else float("nan")

    gamma = main["slope"]
    gain = float(np.exp(yf[-1] - yf[0])) if yf.size else float("nan")
    r2 = main["r_squared"]
    if not (np.isfinite(r2) and r2 >= min_r2):
        reasons.append(f"R2={r2:.3f} < {min_r2}")
    if not (np.isfinite(gamma) and gamma > 0):
        reasons.append(f"gamma={gamma:.4g} is not growth")
    if not (np.isfinite(gain) and gain >= min_gain):
        reasons.append(f"amplitude gain x{gain:.2f} < x{min_gain}")
    if max_onset is not None and t[0] > max_onset:
        reasons.append(f"series starts at t={t[0]:.3g} > {max_onset}: "
                       "the linear phase may predate the data")
    fit_ln = gamma * tf + main["intercept"]
    # Window-free cross-check of the same linear phase; never overrides gamma.
    logistic = fit_logistic_saturation(t, np.exp(y))
    return {
        **logistic,
        "gamma": gamma,
        "gamma_stderr": stderr,
        "gamma_hac_stderr": hac, "gamma_hac_lags": hac_lags,
        "gamma_window_spread": spread,
        "gamma_err": gamma_err,
        "intercept": main["intercept"],
        "r_squared": r2,
        "rvalue": float(np.sqrt(max(r2, 0.0))) if np.isfinite(r2) else float("nan"),
        "n_points": int(tf.size),
        "fit_time_range": (float(tf[0]), float(tf[-1])),
        "linear_phase_start": float(tf[0]),
        "linear_phase_end": float(tf[-1]),
        "amplitude_gain": gain,
        "series_start": float(t[0]),
        "window_source": source,
        "fit_ok": int(not reasons),
        "fit_reject_reason": "; ".join(reasons),
        "time": t,
        "ln_amplitude": y,
        "fit_time": tf,
        "fit_ln_amplitude": fit_ln,
        "fit_amplitude": np.exp(fit_ln),
    }


#: The series of growth_rate_summary.csv whose gamma is quoted for a run, in
#: order of preference: the dominant Fourier mode of dB, then (summaries made
#: before it existed) the domain rms of the vector fluctuation.
REFERENCE_SERIES = ("mode", "total")


#: Geometric branch -> growth_rate_summary.csv series of its strongest mode
#: (physical_diagnostics.classify_mode). With T_perp > T_par ions the
#: compressive oblique branch is the mirror mode and the transverse parallel
#: one the ion-cyclotron wave; both can grow in the same run, so the thesis
#: quotes the mirror gamma from its own series, not from the dominant mode.
BRANCH_SERIES = {"compressive_oblique": "mode_compressive",
                 "transverse_parallel": "mode_transverse"}


def branch_growth_rows(rows: list[dict]) -> dict[str, dict]:
    """The growth_rate_summary.csv row of each branch present, keyed by branch."""
    by_series = {r.get("series"): r for r in rows}
    return {branch: by_series[series] for branch, series in BRANCH_SERIES.items()
            if series in by_series}


def reference_growth_row(rows: list[dict]) -> dict | None:
    """The growth_rate_summary.csv row that holds the growth rate of a run.

    Every consumer (comparisons, convergence study, kappa evolution) goes
    through here, so a thesis table never mixes the mode fit of one run with
    the rms fit of another. A failed reference fit is returned as such: the
    rms series is not a silent fallback, because it is biased low.
    """
    by_series = {r.get("series") or "total": r for r in rows}
    for series in REFERENCE_SERIES:
        if series in by_series:
            return by_series[series]
    return rows[0] if rows else None
