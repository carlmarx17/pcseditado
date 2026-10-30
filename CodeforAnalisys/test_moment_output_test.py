"""Checks for diagnostic discrimination and spatial centring order."""
import numpy as np
import h5py
from moment_output_test import compare_models, read_profiles


def test_discriminates_density_weighted_from_temperature():
    n = np.linspace(0.5, 1.5, 64)
    models = {'pressure': n * .01, 'temperature': np.full(64, .01)}
    for name, data in models.items():
        assert compare_models(data, models)['verdict'] == name


def test_degenerate_or_invalid_data_are_inconclusive():
    for observed, models in [
        (np.ones(8), {'a': np.ones(8), 'b': np.ones(8)}),
        (np.full(8, 100.), {'a': np.ones(8), 'b': np.full(8, 2.)}),
        (np.full(8, np.nan), {'a': np.ones(8), 'b': np.full(8, 2.)}),
    ]:
        assert compare_models(observed, models)['verdict'] == 'inconclusive'


def test_centre_before_averaging_opposite_flows(tmp_path):
    # Each cell has n=2, m=4, T=.5. Opposite flows cancel on averaging,
    # but their kinetic energy must not be mistaken for thermal energy.
    shape = (3, 2, 1)
    n, m, temperature = 2., 4., .5
    velocity = np.broadcast_to(np.array([1., -1.])[None, :, None], shape)
    with h5py.File(tmp_path / 'pfd_moments.000000_p000000.h5', 'w') as f:
        f['crd[2]/crd[2]/p0/1d'] = np.arange(3.)
        for key in ('rho', 'px', 'py', 'pz', 'txx', 'tyy', 'tzz'):
            data = np.full(shape, n if key == 'rho' else 0.)
            if key == 'py':
                data = n*m*velocity
            if key in ('txx', 'tyy', 'tzz'):
                data = np.full(shape, n*temperature)
            if key == 'tyy':
                data = data + n*m*velocity**2
            f[f'all_1st_cc/{key}_i/p0/3d'] = data
    _, profile = read_profiles(tmp_path, 0, mass=m)
    np.testing.assert_allclose(profile['py'], 0.)
    np.testing.assert_allclose(profile['Pyy']/profile['rho'], temperature)
    assert np.all(profile['tyy']/profile['rho'] > temperature)
