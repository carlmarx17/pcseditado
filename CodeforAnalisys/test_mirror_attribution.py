import numpy as np
import pytest

from mirror_attribution import attribution, mode_coefficients


@pytest.mark.parametrize("transverse", [True, False])
def test_periodic_centre_contrast_and_wave_reconstruction(transverse):
    nz, ny = 24, 16
    z = np.arange(nz)[:, None] * 2*np.pi/nz
    ones = np.ones((nz, ny))
    bx = 0.3*np.sin(z)*ones if transverse else ones*0
    bz = ones if transverse else (1-0.1*np.cos(z))*ones
    bmag = np.hypot(bx, bz)
    mask = np.zeros_like(ones, dtype=bool)
    mask[[0, 1, -1]] = True
    maps = {"db": bmag-bmag.mean(), "hole_mask": mask, "peak_mask": ~ones.astype(bool),
            "hole_labels": mask.astype(int)}
    obj = {"label": 1, "kind": "hole"}
    channels = attribution({"x": bx, "y": 0*ones, "z": bz}, maps, [obj], 0.5, 1.)
    measured = obj["contrast_B_over_B0"]
    assert measured < 0
    assert obj["exterior_cells"] == 2*ny
    assert measured == pytest.approx(obj["contrast_Bz_over_B0"] + obj["contrast_transverse_over_B0"])
    assert obj["contrast_fundamental_only_over_B0"] == pytest.approx(measured if transverse else 0)
    coeffs = mode_coefficients(channels, 3)
    peak = max(coeffs, key=lambda r: abs(complex(r["B_over_B0_re"], r["B_over_B0_im"])))
    assert (peak["n_parallel"], peak["n_perp_signed"]) == (2 if transverse else 1, 0)
