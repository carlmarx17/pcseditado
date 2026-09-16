"""Characterize discrete spatial modes before projection or de-growth.

Convention: B = Re[b exp(i k.x - i omega t)], right-handed transverse basis
relative to the supplied background-field axis. No instability name is inferred
from a preset. Fits describe only the selected time interval and plane.
"""
from __future__ import annotations

import numpy as np


def characterize_modes(spectra, times, k0, k1, axes, parallel_axis,
                       component_axes=None, max_modes=12,
                       theta_min_deg=None, theta_max_deg=None):
    """Rank distinct conjugate spatial pairs by time-averaged measured power.

    Each spectrum is complex (time, k0, k1). A single complex exponential is
    fitted per candidate, so beating/phase changes are flagged by coherence
    rather than assigned a precise eigenfrequency. A mode pair counts once.
    Power fractions refer to ALL retained nonzero spatial modes, including
    modes outside an optional angular band. Fits do not establish causality or
    distinguish plasma eigenmodes from advected structures.
    """
    t = np.asarray(times, dtype=float)
    nyquist = float(np.pi / np.median(np.diff(t)))
    spatial_power = sum(np.mean(np.abs(s) ** 2, axis=0) for s in spectra)
    a, b = np.meshgrid(k0, k1, indexing="ij")
    kp, kt = (a, b) if axes[0] == parallel_axis else (b, a)
    nonzero = (np.abs(kp) + np.abs(kt)) > 1e-12
    spatial_power = np.where(nonzero, spatial_power, 0.0)
    total = float(spatial_power.sum())
    if total <= 0:
        return []
    # The retained FFT block is symmetric around k=0 (Nyquist excluded).
    pair_power = spatial_power + spatial_power[::-1, ::-1]
    canonical = (kp > 1e-12) | ((np.abs(kp) < 1e-12) & (kt > 1e-12))
    theta = np.degrees(np.arctan2(np.abs(kt), np.abs(kp)))
    if theta_min_deg is not None:
        canonical &= theta >= theta_min_deg
    if theta_max_deg is not None:
        canonical &= theta <= theta_max_deg
    # True spatial maxima only; no display smoothing. Conjugate pairing
    # preserves total power when a wave travels in either direction.
    padded = np.pad(pair_power, 1, constant_values=-np.inf)
    local_max = np.ones(pair_power.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            local_max &= pair_power >= padded[1+di:1+di+len(k0), 1+dj:1+dj+len(k1)]
    mask = canonical & local_max & (pair_power >= pair_power.max() * 1e-3)
    candidates = np.argwhere(mask)
    candidates = sorted(candidates, key=lambda ij: pair_power[tuple(ij)], reverse=True)[:max_modes]
    rows = []
    for i, j in candidates:
        series = np.stack([s[:, i, j] for s in spectra])
        energy = np.sum(np.abs(series) ** 2, axis=0)
        valid = energy > max(float(energy.max()) * 1e-12, np.finfo(float).tiny)
        if np.count_nonzero(valid) < 6:
            continue
        fit_t = t[valid]
        tc = fit_t - fit_t.mean()
        series = series[:, valid]
        energy = energy[valid]
        fit_resolution = float(2 * np.pi / (fit_t[-1] - fit_t[0]))
        ref = series[np.argmax(np.mean(np.abs(series) ** 2, axis=1))]
        logamp = 0.5 * np.log(np.maximum(energy, np.finfo(float).tiny))
        gamma, intercept = np.polyfit(tc, logamp, 1)
        residual = logamp - (intercept + gamma * tc)
        variance = float(np.sum((logamp - logamp.mean()) ** 2))
        r2 = float(np.clip(1 - np.sum(residual**2) / max(variance, 1e-30), 0, 1))
        mid = len(fit_t) // 2
        gamma_first = float(np.polyfit(tc[:mid], logamp[:mid], 1)[0])
        gamma_second = float(np.polyfit(tc[mid:], logamp[mid:], 1)[0])
        phase = np.unwrap(np.angle(ref))
        slope, phase0 = np.polyfit(tc, phase, 1)
        omega = float(-slope)
        phase_residual = phase - (phase0 + slope * tc)
        coherence = float(abs(np.mean(np.exp(1j * phase_residual))))
        # A stable phase in Bx alone is insufficient when By/Bz contain
        # different frequencies. Require the same omega in every component
        # carrying at least 5% of this spatial mode's magnetic power.
        component_power = np.mean(np.abs(series) ** 2, axis=1)
        for c, strength in enumerate(component_power):
            if strength >= 0.05 * component_power.sum():
                phasor = series[c] / np.maximum(np.abs(series[c]), np.finfo(float).tiny)
                coherence = min(coherence, float(abs(np.mean(phasor * np.exp(1j * omega * tc)))))
        envelope_ok = float(np.std(residual)) < 0.35
        # These are diagnostic acceptance thresholds, not significance levels.
        # Do not identify a noise maximum merely because argmax always exists.
        contiguous = bool(np.allclose(np.diff(fit_t), np.median(np.diff(t))))
        coherent = bool(coherence >= 0.85 and envelope_ok and contiguous)
        growth_consistent = bool(
            r2 >= 0.8 and abs(gamma) * (fit_t[-1] - fit_t[0]) >= 0.25
            and abs(gamma_first - gamma_second) <= max(0.5 * abs(gamma), 1e-12)
        )
        near_nyquist = abs(omega) >= nyquist - fit_resolution
        resolved = coherent and fit_resolution < abs(omega) < nyquist - fit_resolution
        if not coherent:
            status = "incoherent_or_multiple_modes"
        elif near_nyquist:
            status = "near_temporal_nyquist"
        elif resolved:
            status = "propagating_candidate"
        elif growth_consistent and gamma > 0:
            status = "aperiodic_or_slow_growing_candidate"
        else:
            status = "zero_frequency_or_unresolved"
        compressibility = None
        helicity = None
        if component_axes is not None and set(component_axes) == {"x", "y", "z"}:
            compressibility = float(component_power[component_axes.index(parallel_axis)]
                                    / component_power.sum())
            transverse = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[parallel_axis]
            u, v = [series[component_axes.index(axis)] for axis in transverse]
            denom = float(np.mean(np.abs(u)**2 + np.abs(v)**2))
            if denom > 1e-12 * component_power.sum():
                helicity = float(2 * np.imag(np.mean(u * v.conj())) / denom)
        rows.append({
            "power_rank": len(rows) + 1,
            "k_parallel_d_i": float(kp[i, j]),
            "k_perp_signed_d_i": float(kt[i, j]),
            "k_perp_d_i": float(abs(kt[i, j])),
            "k_d_i": float(np.hypot(kp[i, j], kt[i, j])),
            "theta_deg": float(theta[i, j]),
            "power_fraction": float(pair_power[i, j] / total),
            "omega_over_omega_ci": omega,
            "omega_resolution_over_omega_ci": fit_resolution,
            "fit_start_omega_ci_t": float(fit_t[0]),
            "fit_end_omega_ci_t": float(fit_t[-1]),
            "fit_snapshots": int(len(fit_t)),
            "omega_nyquist_over_omega_ci": nyquist,
            "frequency_resolved": bool(resolved),
            "phase_velocity_parallel_over_va": float(omega / kp[i, j]) if resolved and abs(kp[i, j]) > 1e-12 else None,
            "gamma_over_omega_ci": float(gamma),
            "growth_r_squared": r2,
            "gamma_first_half": gamma_first,
            "gamma_second_half": gamma_second,
            "growth_fit_usable": growth_consistent,
            "phase_coherence": coherence,
            "magnetic_compressibility": compressibility,
            "sigma_b_transverse": helicity,
            "accepted": coherent and not near_nyquist,
            "status": status,
        })
    return rows


def plot_mode_summary(result, output):
    """Measured wavevector power and frequency/growth of coherent candidates."""
    import matplotlib.pyplot as plt
    import plot_style as ps

    rows = result["mode_candidates"]
    fig, (ax, summary) = plt.subplots(1, 2, figsize=(12, 5.6),
                                    gridspec_kw={"width_ratios": [1, 1.15]}, layout="constrained")
    ps.style_axes(ax)
    axes, parallel = result["axes"], result["parallel_axis"]
    raw = result["spatial_mode_power"]
    if axes[0] == parallel:
        kp, kt, power = result["k0"], result["k1"], raw.T
    else:
        kp, kt, power = result["k1"], result["k0"], raw
    norm = power / max(float(power.max()), np.finfo(float).tiny)
    mesh = ax.pcolormesh(kp, kt, np.ma.masked_where(norm < 1e-6, 10*np.log10(np.maximum(norm, 1e-30))),
                         cmap=ps.CMAP_SEQUENTIAL, vmin=-60, vmax=0, shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Relative magnetic power [dB]", fontsize=11)
    ax.set_xlabel(r"$k_\parallel d_i$")
    ax.set_ylabel(r"$k_\perp d_i$ (signed)")
    ax.set_title("(a) Spatial modes", fontsize=15)
    dominant = rows[0] if rows else None
    summary.axis("off")
    summary.set_title("(b) Strongest spatial peak", loc="left", fontsize=15)
    if dominant is None:
        summary.text(0, 0.9, "No spatial candidate detected.\nInspect sampling, noise and the time interval.", va="top", wrap=True)
    else:
        m = dominant
        ax.plot(m["k_parallel_d_i"], m["k_perp_signed_d_i"], "o", ms=11,
                markerfacecolor="none", markeredgecolor="white", markeredgewidth=2)
        freq = f"{m['omega_over_omega_ci']:.4g}" if m["frequency_resolved"] else "not resolved"
        comp = m["magnetic_compressibility"]
        pol = m["sigma_b_transverse"]
        lines = [
            f"Status: {m['status'].replace('_', ' ')}",
            "Single-mode characterization: " + ("candidate accepted" if m["accepted"] else "not confirmed"),
            f"k_parallel d_i = {m['k_parallel_d_i']:.4g}",
            f"|k_perp| d_i = {m['k_perp_d_i']:.4g}; angle = {m['theta_deg']:.1f} deg",
            f"Share of retained magnetic power: {100*m['power_fraction']:.1f}%",
            f"omega / Omega_ci: {freq}",
            f"Frequency resolution: {m['omega_resolution_over_omega_ci']:.3g} Omega_ci",
            f"gamma / Omega_ci = {m['gamma_over_omega_ci']:.4g} (R² = {m['growth_r_squared']:.3f})",
            "Growth fit: " + ("consistent across interval" if m["growth_fit_usable"] else "not established"),
            f"Phase coherence = {m['phase_coherence']:.3f}",
        ]
        if comp is not None:
            lines.append(f"Magnetic compressibility = {comp:.3f}")
        if pol is not None:
            lines.append(f"Transverse sigma_B = {pol:+.3f}")
        summary.text(0, 0.96, "\n".join(lines), va="top", fontsize=11, linespacing=1.8)
    times = result["times"]
    fig.supxlabel(f"Selected interval: {times[0]:.3g}–{times[-1]:.3g} Ωci t; "
                  f"{len(times)} snapshots. Frequencies are in the simulation frame.", fontsize=10)
    ps.save(fig, output)


def plot_mode_fit(result, output):
    """Show the actual amplitude and phase used in the strongest peak's fits."""
    import matplotlib.pyplot as plt
    import plot_style as ps

    trace = result["dominant_trace"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    for ax in axes:
        ps.style_axes(ax)
    if trace is None:
        axes[0].text(0.5, 0.5, "No measurable spatial peak", transform=axes[0].transAxes, ha="center")
    else:
        m = result["mode_candidates"][0]
        t, amp, phase = trace["times"], trace["amplitude"], trace["phase"]
        tc = t - t.mean()
        amp = amp / amp.max()
        fit_amp = np.exp(np.mean(np.log(amp)) + m["gamma_over_omega_ci"] * tc)
        fit_phase = phase.mean() - m["omega_over_omega_ci"] * tc
        axes[0].semilogy(t, amp, "o-", ms=3, label="Measured amplitude")
        axes[0].semilogy(t, fit_amp, "--", color=ps.c("#ff7b72"), label="Exponential fit")
        axes[0].set_title("(a) Growth fit: " + ("consistent" if m["growth_fit_usable"] else "not established"), fontsize=13)
        axes[1].plot(t, phase, "o-", ms=3, label="Measured phase (strongest component)")
        axes[1].plot(t, fit_phase, "--", color=ps.c("#ff7b72"), label="Constant-frequency fit")
        axes[1].set_title("(b) Frequency: " + ("resolved candidate" if m["frequency_resolved"] else "not resolved"), fontsize=13)
        for ax in axes:
            ps.legend(ax, fontsize=10)
        fig.suptitle(rf"Strongest spatial peak: $k_\parallel d_i={m['k_parallel_d_i']:.3g}$, "
                     rf"$|k_\perp|d_i={m['k_perp_d_i']:.3g}$", fontsize=14)
    axes[0].set_ylabel("Amplitude / maximum")
    axes[1].set_ylabel("Unwrapped phase [rad]")
    axes[1].set_xlabel(r"$\Omega_{ci}t$")
    ps.save(fig, output)
