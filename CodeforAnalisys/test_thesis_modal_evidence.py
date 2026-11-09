import numpy as np
import pytest

from thesis_modal_evidence import complex_fit, snapshot_budget


def test_growth_and_frequency_after_doppler_shift():
    t = np.linspace(3, 33, 180)
    phase_advection = 0.2*t + 0.001*t**2
    z = np.exp((0.08-0.4j)*t-1j*phase_advection)
    fit = complex_fit(t, z*np.exp(1j*phase_advection))
    assert fit["omega_plasma_fit"] == pytest.approx(.4)
    assert fit["gamma_Bz"] == pytest.approx(.08)
    assert fit["frequency_bin"] == pytest.approx(2*np.pi/30)


def test_global_work_requires_overlap_and_moments_sum_once(tmp_path):
    reservoirs = ["E_B", "E_E", *(f"{k}_{s}" for s in "ie" for k in
                                  ["bulk", "thermal_parallel", "thermal_perp"])]
    data = np.ones(4, dtype=[(k, "f8") for k in ["omega_ci_t", "total_moment_approx", *reservoirs]])
    data["omega_ci_t"] = np.arange(4)
    data["E_B"] += data["omega_ci_t"]
    data["total_moment_approx"] = sum(data[k] for k in reservoirs)
    diag = tmp_path/"energy.csv"
    diag.write_text("omega_ci_t,E_total\n0,80\n1,90\n2,100\n")
    rows = snapshot_budget(data, {"start": 1, "end": 2}, diag)
    assert rows[2]["exact_residual_over_initial"] == rows[2]["moment_residual_over_initial"] == .25
    assert rows[-1]["moment_residual_density"] == 3
    assert "exact_residual_over_initial" not in rows[-1]


def test_tensor_work_includes_out_of_plane_shear_and_periodic_sign():
    from heating_channels import pressure_strain
    n = 32
    y = np.arange(n)*2*np.pi/n
    vx = np.broadcast_to(np.sin(y), (n,n))
    zero = np.zeros_like(vx)
    pxy = np.broadcast_to(-np.cos(y), vx.shape)
    tensor = dict(Vx=vx, Vy=zero, Vz=zero, Pxy=pxy, Pzx=zero, Pyy=zero, Pyz=zero, Pzz=zero)
    assert pressure_strain(tensor, 2*np.pi/n, 1) == pytest.approx(.5*np.sinc(2/n))


def test_field_work_is_transfer_not_an_extra_reservoir(tmp_path):
    from heating_channels import heating_figure
    import json
    phys = tmp_path/"09_physical_diagnostics"
    phys.mkdir()
    (phys/"energy_exchange_summary.json").write_text(json.dumps({"volume_from_E_B0":2}))
    (phys/"energy_exchange_table.csv").write_text(
        "step,omega_ci_t,time_code,W_total_i,W_total_e\n0,0,0,0,0\n1,1,1,0,1\n")
    (phys/"global_energy_table.csv").write_text(
        "time_code,E_E,E_B,E_e,E_i,E_total\n0,4,2,5,7,18\n1,2,2,7,7,18\n")
    assert heating_figure(tmp_path,tmp_path/"review")
    d = np.genfromtxt(tmp_path/"review/heating_channels.csv", names=True, delimiter=",")
    assert d["delta_E_e"][-1] == d["W_e"][-1] == 1
    assert d["EM_closure_residual"][-1] == d["delta_E_total"][-1] == 0
