"""Numerical equivalence for bounded particle reads and reused FFTs."""
import h5py
import numpy as np
import pytest

from data_reader import PICDataReader
from spectral_analysis import SpectralAnalyzer


@pytest.mark.parametrize('limit', [0, 73, 65537, 150000])
@pytest.mark.parametrize('weighted', [False, True])
def test_block_reads_preserve_original_sample(tmp_path, limit, weighted):
    names = ['x', 'y', 'z', 'q', 'm', 'px', 'py', 'pz'] + (['w'] if weighted else [])
    rng = np.random.default_rng(42)
    records = np.empty(131079, dtype=[(name, 'f4') for name in names])
    for name in names:
        records[name] = rng.normal(size=len(records))
    path = tmp_path / 'particles.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('particles/p0/1d', data=records, chunks=True, compression='gzip')
    idx = (np.sort(np.random.default_rng(20260623).choice(len(records), limit, replace=False))
           if len(records) > limit else slice(None))
    spatial = PICDataReader.read_particles_with_positions(str(path), limit)
    for name in names:
        np.testing.assert_array_equal(spatial[name], records[name][idx].astype(float))
    for name, values in zip(('q', 'm', 'px', 'py', 'pz', 'w'),
                            PICDataReader.read_particles_snapshot(str(path), limit)):
        expected = records[name][idx] if name in names else np.ones(len(records[idx]))
        np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_reused_fft_matches_original_power_exactly(tmp_path, axis):
    analyzer = SpectralAnalyzer(dx=1., dy=1., dz=1., outdir=tmp_path, parallel_axis=axis)
    rng = np.random.default_rng(7)
    fluctuations = {f'b{a}': rng.normal(size=(31, 48)) for a in 'xyz'}
    expected = analyzer._component_psds(fluctuations)
    transforms = {a: analyzer._compute_fft_complex(fluctuations[f'b{a}']) for a in 'xyz'}
    actual = analyzer._component_psds(fluctuations, transforms)
    for component in expected:
        np.testing.assert_array_equal(actual[component], expected[component])
