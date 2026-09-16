"""End-to-end regressions for measured modes, independent of display choices."""
import numpy as np
import pytest

from dispersion_analysis import compute_phase_velocity_density, extract_ridges


def signal(omega=0.8, gamma=0.0, mode=(2, 3), nt=128):
    t = np.arange(nt) * 0.5
    y, z = np.meshgrid(np.arange(24) * 0.5, np.arange(32) * 0.5, indexing="ij")
    ky, kz = 2 * np.pi * np.asarray(mode) / np.array([12, 16])
    phase = ky * y + kz * z - omega * t[:, None, None]
    return t, np.exp(gamma * t[:, None, None]) * np.cos(phase), (ky, kz)


def analyse(wave, t, **kwargs):
    return compute_phase_velocity_density(
        wave[None] if wave.ndim == 3 else wave, t, (0.5, 0.5), ("y", "z"),
        max_spatial_mode=10, **kwargs,
    )


def test_isolated_mode_does_not_create_neighbouring_wavenumbers():
    t, wave, (_, kz) = signal()
    rows = extract_ridges(analyse(wave, t), ridge_count=1)
    assert len(rows) == 1
    assert rows[0]["k_parallel_d_i"] == pytest.approx(kz)


def test_signed_velocity_uses_kx_minus_omega_t_convention():
    t, wave, (_, kz) = signal()
    rows = extract_ridges(analyse(wave, t, absolute_velocity=False), ridge_count=1)
    strongest = max(rows, key=lambda row: row["spectral_power"])
    assert strongest["k_parallel_d_i"] == pytest.approx(kz)
    assert strongest["phase_velocity_over_va"] == pytest.approx(0.8 / kz, abs=0.03)


def test_default_keeps_aperiodic_peak():
    t, wave, _ = signal(omega=0, gamma=0.04)
    rows = extract_ridges(analyse(wave, t, degrowth="per-k"), ridge_count=1)
    strongest = max(rows, key=lambda row: row["spectral_power"])
    assert strongest["omega_over_omega_ci"] == 0


def test_padding_does_not_improve_reported_physical_resolution():
    t, wave, _ = signal()
    values = []
    for nfft in (128, 1024):
        rows = extract_ridges(analyse(wave, t, temporal_fft_size=nfft), ridge_count=1)
        values.append(max(rows, key=lambda r: r["spectral_power"])["omega_resolution_over_omega_ci"])
    assert values == pytest.approx([2 * np.pi / (t[-1] - t[0])] * 2)


def test_static_and_zero_fields_are_valid_inputs():
    t, wave, _ = signal(omega=0)
    result = analyse(wave, t, temporal_window="none")
    assert result["power"].max() > 0
    empty = analyse(np.zeros_like(wave), t)
    assert extract_ridges(empty) == []


@pytest.mark.parametrize("bad_times", [np.zeros(128), -np.arange(128), np.full(128, np.nan)])
def test_invalid_times_are_rejected(bad_times):
    _, wave, _ = signal()
    with pytest.raises(ValueError, match="times"):
        analyse(wave, bad_times)


@pytest.mark.parametrize("omega", [0.0, 0.8, -0.8])
def test_full_wavevector_frequency_growth_and_compressibility(omega):
    t, wave, (ky, kz) = signal(omega=omega, gamma=0.04)
    fields = np.stack([np.zeros_like(wave), np.zeros_like(wave), wave])
    result = analyse(fields, t, component_axes=("x", "y", "z"))
    mode = result["mode_candidates"][0]
    assert mode["accepted"]
    assert mode["k_parallel_d_i"] == pytest.approx(kz)
    assert mode["k_perp_d_i"] == pytest.approx(ky)
    assert mode["theta_deg"] == pytest.approx(np.degrees(np.arctan2(ky, kz)))
    assert mode["gamma_over_omega_ci"] == pytest.approx(0.04, abs=1e-10)
    assert mode["growth_fit_usable"]
    assert mode["omega_over_omega_ci"] == pytest.approx(omega, abs=1e-10)
    assert mode["magnetic_compressibility"] == pytest.approx(1.0)
    assert mode["power_fraction"] == pytest.approx(1.0)
    assert mode["frequency_resolved"] == (omega != 0)


def test_circular_polarization_convention():
    t, bx, (ky, kz) = signal(gamma=0.02)
    y, z = np.meshgrid(np.arange(24)*0.5, np.arange(32)*0.5, indexing="ij")
    by = np.exp(0.02*t[:, None, None]) * np.sin(ky*y + kz*z - 0.8*t[:, None, None])
    result = analyse(np.stack([bx, by, np.zeros_like(bx)]), t, component_axes=("x", "y", "z"))
    mode = result["mode_candidates"][0]
    assert mode["sigma_b_transverse"] == pytest.approx(1.0)
    assert mode["magnetic_compressibility"] == 0


def test_noise_does_not_get_a_dominant_coherent_mode():
    t, wave, _ = signal()
    noise = np.random.default_rng(2026).normal(size=wave.shape)
    result = analyse(noise, t)
    assert not any(row["accepted"] for row in result["mode_candidates"])


def test_perpendicular_aperiodic_mode_is_not_lost_in_velocity_projection():
    t, wave, (ky, _) = signal(omega=0, gamma=0.04, mode=(3, 0))
    mode = analyse(wave, t)["mode_candidates"][0]
    assert mode["accepted"]
    assert mode["k_parallel_d_i"] == 0
    assert mode["k_perp_d_i"] == pytest.approx(ky)
    assert mode["theta_deg"] == 90
    assert mode["phase_velocity_parallel_over_va"] is None


def test_ranking_uses_measured_power_before_degrowth():
    t, wave1, (_, kz1) = signal(omega=0.8, gamma=0.06, mode=(1, 2))
    _, wave2, _ = signal(omega=0.4, gamma=0, mode=(4, 5))
    rankings = []
    for degrowth in ("none", "per-k"):
        rankings.append(analyse(wave1 + 2*wave2, t, degrowth=degrowth)["mode_candidates"])
    assert rankings[0] == rankings[1]
    assert rankings[0][0]["k_parallel_d_i"] == pytest.approx(kz1)


def test_no_growth_claim_across_growth_then_saturation():
    t, wave, _ = signal()
    wave *= np.exp(0.08 * np.minimum(t, t[len(t)//2]))[:, None, None]
    mode = analyse(wave, t)["mode_candidates"][0]
    assert not mode["growth_fit_usable"]


def test_two_separated_frequencies_in_one_k_bin():
    t, wave1, (_, kz) = signal(omega=0.8)
    _, wave2, _ = signal(omega=1.7)
    rows = extract_ridges(analyse(wave1 + 0.7*wave2, t, temporal_fft_size=1024), ridge_count=2)
    assert len(rows) == 2
    assert [row["k_parallel_d_i"] for row in rows] == pytest.approx([kz, kz])
    assert sorted(row["omega_over_omega_ci"] for row in rows) == pytest.approx([0.8, 1.7], abs=0.03)


def test_zero_initial_snapshot_does_not_bias_growth_fit():
    t, wave, _ = signal(gamma=0.04)
    wave[0] = 0
    result = analyse(wave, t, degrowth="per-k")
    mode = result["mode_candidates"][0]
    assert mode["gamma_over_omega_ci"] == pytest.approx(0.04, abs=1e-10)
    assert result["gamma_global"] == pytest.approx(0.04, abs=1e-10)
    assert mode["fit_snapshots"] == len(t) - 1


def test_kmax_is_a_physical_magnitude_cap():
    t, wave, (ky, kz) = signal()
    result = analyse(wave, t, kmax_di=1.3)
    a, b = np.meshgrid(result["k0"], result["k1"], indexing="ij")
    outside = np.hypot(a, b) > 1.3
    assert np.all(result["power"][:, outside] == 0)


def test_rank_two_is_not_a_duplicate_of_rank_one_in_legacy_walk():
    t, wave, _ = signal()
    rows = extract_ridges(analyse(wave, t), ridge_count=2, ridge_axis="omega")
    keys = [(r["omega_over_omega_ci"], r["k_parallel_d_i"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_wave_is_characterized_on_broadband_noise_floor():
    t, wave, (_, kz) = signal(gamma=0.02)
    wave += np.random.default_rng(19).normal(size=wave.shape)
    mode = analyse(wave, t)["mode_candidates"][0]
    assert mode["accepted"]
    assert mode["k_parallel_d_i"] == pytest.approx(kz)
    assert mode["omega_over_omega_ci"] == pytest.approx(0.8, abs=0.02)
    assert mode["gamma_over_omega_ci"] == pytest.approx(0.02, abs=0.002)


def test_near_nyquist_is_not_confirmed():
    t, wave, _ = signal(omega=2*np.pi - 0.02)
    mode = analyse(wave, t)["mode_candidates"][0]
    assert not mode["frequency_resolved"]
    assert not mode["accepted"]
    assert mode["status"] == "near_temporal_nyquist"


def test_different_frequencies_in_components_are_not_one_mode():
    t, bx, _ = signal(omega=0.8)
    _, by, _ = signal(omega=1.7)
    mode = analyse(np.stack([bx, by, np.zeros_like(bx)]), t,
                   component_axes=("x", "y", "z"))["mode_candidates"][0]
    assert not mode["accepted"]


def test_window_sidelobes_are_not_reported_as_a_second_branch():
    t, wave, _ = signal()
    rows = extract_ridges(analyse(wave, t, temporal_fft_size=1024), ridge_count=2)
    assert len(rows) == 1


def test_exports_empty_and_perpendicular_only_spectra(tmp_path):
    from dispersion_analysis import plot_omega_k_dispersion
    from dispersion_modes import plot_mode_summary, plot_mode_fit
    t, wave, _ = signal(omega=0, mode=(3, 0))
    for label, values, cap in [("empty", np.zeros_like(wave), 0.1), ("perpendicular", wave, None)]:
        result = analyse(values, t, kmax_di=cap)
        for name, function in [("summary", plot_mode_summary), ("fit", plot_mode_fit)]:
            path = tmp_path / f"{label}_{name}.png"
            function(result, path)
            assert path.exists()
            assert path.with_suffix(".pdf").exists()
        plot_omega_k_dispersion(result, tmp_path / f"{label}_wk.png", "total", 0.01)
