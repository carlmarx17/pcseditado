#!/usr/bin/env python3
"""Synthetic-signal checks for the omega-k pipeline.

Each test builds a field whose answer is known analytically and asks the
pipeline to recover it. The two that matter are:

  * a propagating wave with omega_r != 0   -> the peak must land on omega_r
  * a purely growing mode with omega_r = 0 -> the peak must land on omega = 0

The second one is the regression test for the bug this suite was written for:
subtracting the per-cell time mean deletes the omega = 0 row exactly, so an
aperiodic mode is reported at a finite, k-independent frequency set by the
analysis window rather than by the physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from dispersion_analysis import (
    _fit_growth_per_mode,
    _smooth_2d,
    compute_phase_velocity_density,
    extract_ridges,
    resolve_mode_expectations,
    spectral_resolution_report,
)

SPACING = (0.25, 0.25)          # d_i per cell
AXES = ("y", "z")
N_CELLS = 48                    # box = 12 d_i, dk d_i = 0.5236
N_TIME = 256


def _times(t_max: float = 60.0) -> np.ndarray:
    return np.linspace(0.0, t_max, N_TIME)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(N_CELLS) * SPACING[0]
    z = np.arange(N_CELLS) * SPACING[1]
    return np.meshgrid(y, z, indexing="ij")


def _peak_omega(result: dict) -> float:
    """Frequency of the global power maximum over omega >= 0."""
    omega = result["omega"]
    power = result["power"].sum(axis=(1, 2))
    keep = omega >= 0
    return float(omega[keep][np.argmax(power[keep])])


def _analyse(field: np.ndarray, times: np.ndarray, **kwargs) -> dict:
    defaults = dict(
        parallel_axis="z",
        max_spatial_mode=12,
        temporal_fft_size=N_TIME,
        velocity_min=0.0,
        velocity_max=50.0,
    )
    defaults.update(kwargs)
    return compute_phase_velocity_density(
        field[None, :, :, :], times, SPACING, AXES, **defaults
    )


def test_propagating_wave_recovers_its_frequency():
    """A clean travelling wave must be reported at its own omega_r."""
    times = _times()
    _, zz = _grid()
    k_par = 4 * 2.0 * np.pi / (N_CELLS * SPACING[1])   # k d_i ~ 2.09
    omega_r = 0.8
    field = np.cos(k_par * zz[None, :, :] - omega_r * times[:, None, None])

    result = _analyse(field, times)
    domega = float(result["omega"][1] - result["omega"][0])
    assert _peak_omega(result) == pytest.approx(omega_r, abs=3 * domega)


def test_aperiodic_mode_peaks_at_zero_frequency():
    """A purely growing mode has omega_r = 0 and must be reported there."""
    times = _times()
    _, zz = _grid()
    k_par = 3 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    gamma = 0.05
    field = (
        np.exp(gamma * times)[:, None, None]
        * np.cos(k_par * zz[None, :, :])
    )

    result = _analyse(field, times, time_detrend="none", degrowth="per-k")
    domega = float(result["omega"][1] - result["omega"][0])
    assert abs(_peak_omega(result)) <= 2 * domega


def test_mean_detrending_moves_an_aperiodic_peak_off_zero():
    """Regression guard: the old hard-wired detrending relocates the peak.

    This is not a desirable behaviour being pinned down, it is the failure the
    default was changed to avoid. If a future refactor makes 'mean' harmless
    this test should be revisited, not silenced.
    """
    times = _times()
    _, zz = _grid()
    k_par = 3 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    field = (
        np.exp(0.05 * times)[:, None, None]
        * np.cos(k_par * zz[None, :, :])
    )

    kept = _analyse(field, times, time_detrend="none", degrowth="per-k")
    removed = _analyse(field, times, time_detrend="mean", degrowth="per-k")
    domega = float(kept["omega"][1] - kept["omega"][0])

    # The mode sits at omega = 0; with the row intact the peak is there, and
    # with it removed the peak is pushed to a finite frequency set by the
    # window, not by the physics.
    assert abs(_peak_omega(kept)) < domega
    assert abs(_peak_omega(removed)) >= domega


def test_degrowth_recovers_the_imposed_growth_rate():
    times = _times()
    _, zz = _grid()
    k_par = 3 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    gamma = 0.07
    field = (
        np.exp(gamma * times)[:, None, None]
        * np.cos(k_par * zz[None, :, :] - 0.4 * times[:, None, None])
    )

    result = _analyse(field, times, degrowth="global")
    assert result["gamma_global"] == pytest.approx(gamma, rel=0.15)


def test_growth_fit_ignores_dead_modes():
    """Modes with no power must get gamma = 0 rather than a noise-fitted slope."""
    times = _times()
    power = np.zeros((N_TIME, 3, 3))
    power[:, 1, 1] = np.exp(2 * 0.1 * times)
    gamma, r_squared = _fit_growth_per_mode(power, times)

    assert gamma[1, 1] == pytest.approx(0.1, rel=1e-6)
    assert r_squared[1, 1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(gamma[np.arange(3) != 1] == 0.0)


def test_ridge_axis_k_gives_one_row_per_wavenumber():
    """With ridge_axis='k' the CSV is a dispersion relation, not a stripe."""
    times = _times()
    _, zz = _grid()
    k_par = 4 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    field = np.cos(k_par * zz[None, :, :] - 0.8 * times[:, None, None])

    result = _analyse(field, times)
    rows = extract_ridges(result, ridge_count=1, ridge_axis="k")

    wavenumbers = [row["k_parallel_d_i"] for row in rows]
    assert len(wavenumbers) == len(set(wavenumbers))
    assert all("omega_fwhm_over_omega_ci" in row for row in rows)


def _noisy_wave(k_par: float, omega_r: float, noise: float, seed: int = 7):
    """A travelling wave on a broadband floor, as PIC data actually arrives."""
    times = _times()
    _, zz = _grid()
    rng = np.random.default_rng(seed)
    field = (
        np.cos(k_par * zz[None, :, :] - omega_r * times[:, None, None])
        + noise * rng.standard_normal((N_TIME, N_CELLS, N_CELLS))
    )
    return field, times


def test_jacobian_bias_pins_ridges_to_small_wavenumbers():
    """The |omega|/k^2 weight is a 1/k^2 bias on the native grid, not a fix.

    Without a broadband floor the weight is harmless, because a noiseless wave
    dominates every column by many decades. With particle noise present -- the
    only regime that matters for PIC output -- it drags the tracked ridge down
    towards the smallest resolved wavenumber.
    """
    k_par = 8 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    field, times = _noisy_wave(k_par, omega_r=1.0, noise=1.0)
    result = _analyse(field, times)

    biased = extract_ridges(
        result, ridge_count=1, ridge_axis="omega", apply_jacobian=True
    )
    unbiased = extract_ridges(
        result, ridge_count=1, ridge_axis="omega", apply_jacobian=False
    )
    biased_k = float(np.median([row["k_parallel_d_i"] for row in biased]))
    unbiased_k = float(np.median([row["k_parallel_d_i"] for row in unbiased]))

    assert biased_k < unbiased_k


def test_ridge_axis_k_survives_a_broadband_noise_floor():
    """Scanning k rather than omega isolates the wave even at SNR ~ 1."""
    k_par = 8 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    omega_r = 1.0
    field, times = _noisy_wave(k_par, omega_r, noise=1.0)
    result = _analyse(field, times)

    rows = extract_ridges(result, ridge_count=1, ridge_axis="k")
    strongest = max(rows, key=lambda row: row["spectral_power"])
    domega = float(result["omega"][1] - result["omega"][0])

    assert strongest["k_parallel_d_i"] == pytest.approx(k_par, abs=1e-9)
    assert strongest["omega_over_omega_ci"] == pytest.approx(omega_r, abs=3 * domega)
    assert strongest["resolved"] == 1


def test_kmax_di_shrinks_the_retained_band():
    times = _times()
    _, zz = _grid()
    field = np.cos(2.0 * zz[None, :, :] - 0.5 * times[:, None, None])

    wide = _analyse(field, times, max_spatial_mode=20)
    narrow = _analyse(field, times, max_spatial_mode=20, kmax_di=2.0)
    assert narrow["power"].shape[1] < wide["power"].shape[1]
    assert float(np.max(narrow["k1"])) < float(np.max(wide["k1"])) + 1e-9


def test_spatial_window_scatters_power_off_an_exact_box_mode():
    """A periodic record needs no spatial window, and paying for one costs a third
    of the mode.

    Hann is a multiplication in x, hence a 3-point convolution in k with weights
    (-1/4, 1/2, -1/4) on the amplitude. An exact box mode therefore keeps 2/3 of
    its power and donates 1/6 to each neighbour -- deterministic, not an artefact
    of this particular signal.
    """
    n = 128
    mode = 6
    signal = np.cos(2.0 * np.pi * mode * np.arange(n) / n)

    rectangular = np.abs(np.fft.rfft(signal)) ** 2
    rectangular /= rectangular.sum()
    hann = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2
    hann /= hann.sum()

    assert rectangular[mode] == pytest.approx(1.0, abs=1e-12)
    assert hann[mode] == pytest.approx(2 / 3, abs=0.01)
    assert hann[mode - 1] == pytest.approx(1 / 6, abs=0.01)
    assert hann[mode + 1] == pytest.approx(1 / 6, abs=0.01)


def test_no_spatial_window_keeps_a_box_mode_in_one_bin():
    """End to end: the default must not spread a single mode across neighbours."""
    times = _times()
    _, zz = _grid()
    k_par = 4 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    field = np.cos(k_par * zz[None, :, :] - 0.8 * times[:, None, None])

    unwindowed = _analyse(field, times, spatial_window="none")
    windowed = _analyse(field, times, spatial_window="hann")

    def concentration(result):
        # Collapse omega and k_perp. A real field puts equal power at +k and -k,
        # so the conjugate pair is what "one mode" means here.
        power = result["power"].sum(axis=0).sum(axis=0)
        pair = np.sort(power)[-2:].sum()
        return float(pair / np.sum(power))

    assert concentration(unwindowed) > concentration(windowed)
    assert concentration(unwindowed) > 0.99


def test_tukey_window_interpolates_between_rectangular_and_hann():
    from dispersion_analysis import _make_window

    n = 256
    assert _make_window("none", n) == pytest.approx(np.ones(n))
    assert _make_window("tukey", n, alpha=0.0) == pytest.approx(np.ones(n))
    # alpha = 1 tapers the whole record, i.e. a Hann window
    assert _make_window("tukey", n, alpha=1.0) == pytest.approx(
        np.hanning(n), abs=0.02
    )
    tukey = _make_window("tukey", n, alpha=0.25)
    assert tukey[n // 2] == pytest.approx(1.0)      # flat in the middle
    assert tukey[0] == pytest.approx(0.0, abs=1e-9)  # tapered at the ends


def test_smoothing_does_not_suppress_the_boundary_row():
    """Zero-padded smoothing would eat the omega = 0 row, where aperiodic modes live."""
    flat = np.ones((9, 9))
    assert _smooth_2d(flat, passes=2)[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_subbin_interpolation_beats_the_frequency_grid():
    """omega between two bins must be recovered better than half a bin."""
    times = _times()
    _, zz = _grid()
    k_par = 6 * 2.0 * np.pi / (N_CELLS * SPACING[1])
    domega = 2.0 * np.pi / (times[-1] - times[0])
    omega_r = 3.5 * domega           # deliberately half-way between bins
    field = np.cos(k_par * zz[None, :, :] - omega_r * times[:, None, None])

    result = _analyse(field, times)
    rows = extract_ridges(result, ridge_count=1, ridge_axis="k")
    strongest = max(rows, key=lambda row: row["spectral_power"])

    grid_omega = result["omega"]
    nearest_bin = float(
        grid_omega[np.argmin(np.abs(grid_omega - omega_r))]
    )
    error = abs(strongest["omega_over_omega_ci"] - omega_r)
    assert error < abs(nearest_bin - omega_r) or error < 0.05 * domega


def test_resolution_report_flags_an_undersampled_box():
    report = spectral_resolution_report(
        _times(), SPACING, (N_CELLS, N_CELLS), mode="firehose-parallel",
        gamma_ci=0.05,
    )
    statuses = {entry["name"]: entry["status"] for entry in report["checks"]}
    # box = 12 d_i -> dk d_i = 0.524, so the 0.2-0.6 band holds 1 mode
    assert statuses["k_sampling"] == "WARN"
    assert report["modes_in_expected_k_band"] < 8


def test_resolution_report_calls_out_aperiodic_modes():
    report = spectral_resolution_report(
        _times(), SPACING, (N_CELLS, N_CELLS), mode="mirror", gamma_ci=0.02,
    )
    branch = [c for c in report["checks"] if c["name"] == "omega_branch_exists"]
    assert branch and branch[0]["status"] == "WARN"


def test_whistler_preset_is_rescaled_to_ion_units():
    preset = resolve_mode_expectations("whistler", mass_ratio=200.0)
    assert preset["omega_r_ci"] == pytest.approx(0.3 * 200.0)
    # k d_i = k d_e * sqrt(mi/me)
    assert preset["k_di"][0] == pytest.approx(0.3 * np.sqrt(200.0))


def test_whistler_preset_requires_a_mass_ratio():
    with pytest.raises(ValueError, match="mass ratio"):
        resolve_mode_expectations("whistler", mass_ratio=None)
