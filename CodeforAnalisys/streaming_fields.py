"""Read one field snapshot at a time; retain only requested Fourier cells."""
import numpy as np


class SnapshotSeries:
    ndim = 4

    def __init__(self, files, loader):
        self.files = list(files)
        self.loader = loader
        first = np.asarray(loader(self.files[0]))
        self.shape = (first.shape[0], len(self.files), *first.shape[1:])

    def snapshots(self):
        expected = (self.shape[0], *self.shape[2:])
        for path in self.files:
            frame = np.asarray(self.loader(path), dtype=float)
            if frame.shape != expected:
                raise ValueError(f"Grid/component shape changed in {path}: {frame.shape} != {expected}")
            yield frame


def snapshots(series):
    if isinstance(series, SnapshotSeries):
        yield from series.snapshots()
    else:
        for i in range(series.shape[1]):
            yield np.asarray(series[:, i], dtype=float)


def spatial_spectra(series, window, slice0, slice1):
    """Spatial demeaning and FFT commute with subsequent temporal detrending.

    Peak workspace scales with one spatial snapshot, not the duration. No
    temporal decimation or spatial resampling is performed.
    """
    nc, nt, _, _ = series.shape
    shape = (nc, nt, slice0.stop - slice0.start, slice1.stop - slice1.start)
    result = np.empty(shape, dtype=np.complex128)
    for t, frame in enumerate(snapshots(series)):
        if not np.all(np.isfinite(frame)):
            raise ValueError("Field snapshots contain NaN or infinity")
        for c, plane in enumerate(frame):
            centered = (plane - plane.mean()) * window
            transformed = np.fft.fftshift(np.fft.fft2(centered))
            result[c, t] = transformed[slice0, slice1]
    return result


def retained_slice(axis, limit):
    if limit is None:
        return slice(0, len(axis))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Wavenumber limit must be finite and positive")
    idx = np.flatnonzero(np.abs(axis) <= limit * (1 + 1e-12))
    return slice(int(idx[0]), int(idx[-1]) + 1)
