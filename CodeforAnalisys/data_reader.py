"""Unified readers for PSC HDF5 and ADIOS2 output."""

from __future__ import annotations

import glob
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import h5py
import numpy as np


class _DataFileView:
    """Small common interface over an HDF5 file or one ADIOS2 step."""

    backend: str

    def keys(self) -> list[str]:
        raise NotImplementedError

    def read(self, path: str) -> np.ndarray:
        raise NotImplementedError


class _HDF5FileView(_DataFileView):
    backend = "hdf5"

    def __init__(self, handle: h5py.File):
        self.handle = handle
        self._dataset_keys: list[str] | None = None

    def keys(self) -> list[str]:
        if self._dataset_keys is None:
            keys: list[str] = []

            def collect(name, node):
                if isinstance(node, h5py.Dataset):
                    keys.append(name)

            self.handle.visititems(collect)
            self._dataset_keys = keys
        return self._dataset_keys

    def read(self, path: str) -> np.ndarray:
        return np.asarray(self.handle[path][()])


class _ADIOS2FileView(_DataFileView):
    backend = "adios2"

    def __init__(self, reader):
        self.reader = reader

    def keys(self) -> list[str]:
        variables = self.reader.available_variables()
        return list(variables.keys())

    def read(self, path: str) -> np.ndarray:
        return np.asarray(self.reader.read(path))


class PICDataReader:
    """Read PSC fields, moments and particles from HDF5 or ADIOS2 BP."""

    STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.(?:h5|bp)/?$")

    @staticmethod
    def is_adios2_path(filename: str | Path) -> bool:
        """Return true only for a path whose logical suffix is ``.bp``."""
        return Path(str(filename).rstrip("/")).suffix.lower() == ".bp"

    @staticmethod
    def get_step_from_filename(filename: str) -> Optional[int]:
        """Extract the PSC step from an HDF5 filename or BP directory."""
        match = PICDataReader.STEP_RE.search(str(filename).rstrip("/"))
        return int(match.group(1)) if match else None

    @staticmethod
    def _alternative_patterns(pattern: str) -> list[str]:
        """Generate the equivalent HDF5/BP patterns used by PSC."""
        alternatives = [pattern]
        if pattern.endswith(".h5"):
            path = Path(pattern)
            if (
                "_p" not in path.name
                and (
                    path.name.startswith("pfd.")
                    or path.name.startswith("pfd_moments.")
                )
            ):
                alternatives.append(str(path.with_name(path.stem + "_p*.h5")))
            bp_pattern = pattern[:-3] + ".bp"
            alternatives.append(bp_pattern)
            alternatives.append(re.sub(r"_p\*\.bp$", ".bp", bp_pattern))
            alternatives.append(re.sub(r"_p\d+\.bp$", ".bp", bp_pattern))
        elif pattern.endswith(".bp"):
            h5_pattern = pattern[:-3] + ".h5"
            alternatives.append(h5_pattern)
            if "_p" not in Path(h5_pattern).name and (
                Path(h5_pattern).name.startswith("pfd.")
                or Path(h5_pattern).name.startswith("pfd_moments.")
            ):
                alternatives.append(h5_pattern[:-3] + "_p*.h5")
        return list(dict.fromkeys(alternatives))

    @staticmethod
    def find_files(pattern: str | Iterable[str]) -> Dict[int, str]:
        """Map steps to snapshots, searching equivalent ``.h5`` and ``.bp`` patterns."""
        requested = [pattern] if isinstance(pattern, str) else list(pattern)
        patterns: list[str] = []
        for item in requested:
            patterns.extend(PICDataReader._alternative_patterns(str(item)))

        files = sorted(
            {
                matched
                for candidate in patterns
                for matched in glob.glob(candidate)
            }
        )
        file_map: Dict[int, str] = {}
        for filename in files:
            step = PICDataReader.get_step_from_filename(filename)
            if step is None:
                continue
            if step in file_map and Path(file_map[step]) != Path(filename):
                raise ValueError(
                    f"Multiple snapshots found for step {step}: "
                    f"{file_map[step]} and {filename}. Keep one assembled "
                    "HDF5 file or one ADIOS2 BP directory per step."
                )
            file_map[step] = filename
        return dict(sorted(file_map.items()))

    @staticmethod
    def discover_outputs(data_dir: str) -> Dict[str, object]:
        """Discover field, moment and particle series in either supported format."""
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {root}")

        fields = PICDataReader.find_files(
            [str(root / "pfd.*_p*.h5"), str(root / "pfd.*.bp")]
        )
        moments = PICDataReader.find_files(
            [str(root / "pfd_moments.*_p*.h5"), str(root / "pfd_moments.*.bp")]
        )

        particle_series: Dict[str, Dict[int, str]] = {}
        for file_pattern in ("prt*.h5", "prt*.bp"):
            for path in sorted(root.glob(file_pattern)):
                step = PICDataReader.get_step_from_filename(str(path))
                match = re.match(
                    r"(.+?)\.\d+(?:_p\d+)?\.(?:h5|bp)$",
                    path.name.rstrip("/"),
                )
                if step is None or match is None:
                    continue
                series = particle_series.setdefault(match.group(1), {})
                if step in series and Path(series[step]) != path:
                    raise ValueError(
                        f"Particle step {step} exists in both {series[step]} and {path}."
                    )
                series[step] = str(path)

        return {
            "data_dir": root,
            "fields": fields,
            "moments": moments,
            "particles": particle_series,
        }

    @staticmethod
    @contextmanager
    def open_data_file(filename: str | Path) -> Iterator[_DataFileView]:
        """Open one HDF5 snapshot or the first step in an ADIOS2 BP snapshot.

        ADIOS2 has changed its Python high-level API over time. This helper
        supports the current ``FileReader``/``Stream`` APIs and the older
        iterable ``open`` API used by some COSMA installations.
        """
        filepath = str(filename).rstrip("/")
        if not PICDataReader.is_adios2_path(filepath):
            with h5py.File(filepath, "r") as handle:
                yield _HDF5FileView(handle)
            return

        try:
            import adios2
        except ImportError as exc:
            raise ImportError(
                f"Cannot read ADIOS2 snapshot '{filepath}': install the "
                "'adios2' Python bindings or run in the COSMA ADIOS2 environment."
            ) from exc

        if hasattr(adios2, "FileReader"):
            reader = adios2.FileReader(filepath)
            try:
                yield _ADIOS2FileView(reader)
            finally:
                close = getattr(reader, "close", None)
                if close:
                    close()
            return

        if hasattr(adios2, "Stream"):
            with adios2.Stream(filepath, "r") as stream:
                steps = iter(stream.steps())
                try:
                    next(steps)
                except StopIteration as exc:
                    raise ValueError(f"ADIOS2 snapshot has no readable steps: {filepath}") from exc
                yield _ADIOS2FileView(stream)
            return

        if hasattr(adios2, "open"):
            with adios2.open(filepath, "r") as stream:
                try:
                    step = next(iter(stream))
                except StopIteration as exc:
                    raise ValueError(f"ADIOS2 snapshot has no readable steps: {filepath}") from exc
                yield _ADIOS2FileView(step)
            return

        raise RuntimeError("Unsupported ADIOS2 Python API: no FileReader, Stream or open")

    @staticmethod
    def get_uid_group(handle: h5py.File, base_prefix: str) -> str:
        """Compatibility helper for scripts that still inspect HDF5 groups."""
        prefix = base_prefix.rstrip("-/")
        for key in handle.keys():
            if key == prefix or key.startswith(prefix + "-") or key.startswith(prefix + "_"):
                return key
        raise KeyError(f"No group starting with '{base_prefix}' found in file.")

    @staticmethod
    def resolve_dataset_path(
        keys: Iterable[str], group_prefix: str, dataset_path: str
    ) -> Optional[str]:
        """Resolve clean ADIOS2 groups and HDF5 groups with dynamic UID suffixes."""
        prefix = group_prefix.strip("/").rstrip("-")
        suffix = dataset_path.strip("/")
        candidates: list[tuple[int, str]] = []
        for raw_key in keys:
            key = raw_key.strip("/")
            marker = "/" + suffix
            if not key.endswith(marker):
                continue
            group = key[: -len(marker)]
            if group == prefix:
                candidates.append((0, raw_key))
            elif group.startswith(prefix + "-") or group.startswith(prefix + "_"):
                candidates.append((1, raw_key))
        return min(candidates, default=(99, None))[1]

    @staticmethod
    def resolve_variable_path(keys: Iterable[str], variable_path: str) -> Optional[str]:
        """Find a variable by exact logical path, tolerating a leading namespace."""
        wanted = variable_path.strip("/")
        exact = [key for key in keys if key.strip("/") == wanted]
        if exact:
            return exact[0]
        suffix = "/" + wanted
        matches = [key for key in keys if key.strip("/").endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def read_multiple_fields_3d(
        filename: str, group_prefix: str, dataset_paths: List[str]
    ) -> Dict[str, np.ndarray]:
        """Read several fields/moments through the unified backend."""
        results: Dict[str, np.ndarray] = {}
        with PICDataReader.open_data_file(filename) as data_file:
            keys = data_file.keys()
            for dataset_path in dataset_paths:
                resolved = PICDataReader.resolve_dataset_path(
                    keys, group_prefix, dataset_path
                )
                if resolved is None:
                    raise KeyError(
                        f"Dataset '{dataset_path}' with group prefix "
                        f"'{group_prefix}' was not found in {data_file.backend} "
                        f"snapshot '{filename}'."
                    )
                results[dataset_path] = data_file.read(resolved)
        return results

    @staticmethod
    def read_particles_snapshot(
        filepath: str, max_particles: int, rng=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read and uniformly subsample one HDF5 or ADIOS2 particle snapshot."""
        if rng is None:
            rng = np.random.default_rng(20260623)

        if PICDataReader.is_adios2_path(filepath):
            variable_names = ("q", "m", "px", "py", "pz", "w")
            arrays: dict[str, np.ndarray] = {}
            with PICDataReader.open_data_file(filepath) as data_file:
                keys = data_file.keys()
                for name in variable_names:
                    logical_path = f"particles/p0/1d/{name}"
                    resolved = PICDataReader.resolve_variable_path(keys, logical_path)
                    if resolved is None:
                        if name == "w":
                            continue
                        raise KeyError(
                            f"Particle variable '{logical_path}' not found in '{filepath}'."
                        )
                    arrays[name] = np.asarray(data_file.read(resolved))
            q_all = arrays["q"]
            n_total = len(q_all)
            idx = (
                np.sort(rng.choice(n_total, max_particles, replace=False))
                if n_total > max_particles
                else slice(None)
            )
            q = np.asarray(q_all[idx], dtype=float)
            m = np.asarray(arrays["m"][idx], dtype=float)
            px = np.asarray(arrays["px"][idx], dtype=float)
            py = np.asarray(arrays["py"][idx], dtype=float)
            pz = np.asarray(arrays["pz"][idx], dtype=float)
            w = (
                np.asarray(arrays["w"][idx], dtype=float)
                if "w" in arrays
                else np.ones_like(q)
            )
            return q, m, px, py, pz, w

        with h5py.File(filepath, "r") as handle:
            dataset = handle["particles"]["p0"]["1d"]
            n_total = len(dataset)
            idx = (
                np.sort(rng.choice(n_total, max_particles, replace=False))
                if n_total > max_particles
                else slice(None)
            )
            names = dataset.dtype.names or ()
            required = ("q", "m", "px", "py", "pz")
            missing = [name for name in required if name not in names]
            if missing:
                raise KeyError(f"Missing particle fields {missing} in '{filepath}'.")
            q = np.asarray(dataset["q"][idx], dtype=float)
            m = np.asarray(dataset["m"][idx], dtype=float)
            px = np.asarray(dataset["px"][idx], dtype=float)
            py = np.asarray(dataset["py"][idx], dtype=float)
            pz = np.asarray(dataset["pz"][idx], dtype=float)
            w = (
                np.asarray(dataset["w"][idx], dtype=float)
                if "w" in names
                else np.ones_like(q)
            )
        return q, m, px, py, pz, w

    @staticmethod
    def flatten_2d_slice(data_3d: np.ndarray) -> np.ndarray:
        """Remove a singleton dimension from a nominally 2D PSC output."""
        data = np.asarray(data_3d)
        if data.ndim == 3 and data.shape[2] == 1:
            return data[:, :, 0]
        if data.ndim == 3 and data.shape[0] == 1:
            return data[0, :, :]
        return data.squeeze()
