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
4. Inside the rise, the linear phase is the contiguous interval around the
   maximum of the local log-slope where that slope stays >= ``slope_frac``
   (default 80 %) of its maximum. Without this step the band includes the
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
    j = int(np.argmax(slope))
    if not slope[j] > 0:
        return start, end
    keep = slope >= slope_frac * slope[j]
    a = b = j
    while a - 1 >= 0 and keep[a - 1]:
        a -= 1
    while b + 1 < keep.size and keep[b + 1]:
        b += 1
    if b - a + 1 < 4:
        # The slope is too noisy for the criterion to isolate a phase (a lone
        # spike): keep the whole rise, whose bias the sensitivity refits show.
        return start, end
    return start + a, start + b


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
        "gamma": float("nan"), "gamma_stderr": float("nan"),
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
    parts = [v for v in (stderr, spread) if np.isfinite(v)]
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
    return {
        "gamma": gamma,
        "gamma_stderr": stderr,
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
