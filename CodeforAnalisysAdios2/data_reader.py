import adios2
import glob
import re
import numpy as np
from typing import Dict, List, Optional

class PICDataReader:
    """Utility class to read and extract data from PIC simulation ADIOS2 output files."""

    PACKED_FIELD_MAP = {
        "jx_ec/p0/3d": ("jeh", 0),
        "jy_ec/p0/3d": ("jeh", 1),
        "jz_ec/p0/3d": ("jeh", 2),
        "ex_ec/p0/3d": ("jeh", 3),
        "ey_ec/p0/3d": ("jeh", 4),
        "ez_ec/p0/3d": ("jeh", 5),
        "hx_fc/p0/3d": ("jeh", 6),
        "hy_fc/p0/3d": ("jeh", 7),
        "hz_fc/p0/3d": ("jeh", 8),
    }

    _MOMENT_COMPONENTS = [
        "rho", "jx", "jy", "jz", "px", "py", "pz", "txx", "tyy", "tzz", "txy", "tyz", "tzx",
    ]
    @staticmethod
    def _open_reader(filename: str):
        """Open an ADIOS2 BP file using the API available in this environment."""
        return adios2.FileReader(filename)
    
    @staticmethod
    def get_step_from_filename(filename: str) -> Optional[int]:
        """Extracts the step number from a padded string like pfd_moments.001000.bp"""
        match = re.search(r"\.(\d+)\.bp", filename)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def find_files(pattern: str) -> Dict[int, str]:
        """Returns a sorted dictionary mapping step numbers to filenames."""
        files = sorted(glob.glob(pattern))
        file_map = {}
        for f in files:
            step = PICDataReader.get_step_from_filename(f)
            if step is not None:
                file_map[step] = f
        return file_map

    @staticmethod
    def read_multiple_fields_3d(filename: str, group_prefix: str, dataset_paths: List[str]) -> Dict[str, np.ndarray]:
        """Reads multiple 3D datasets from the ADIOS2 file. group_prefix is ignored."""
        results = {}
        f = PICDataReader._open_reader(filename)
        try:
            packed_cache = {}
            for path in dataset_paths:
                try:
                    variable = f.inquire_variable(path)
                    if variable is not None:
                        results[path] = f.read(variable)
                        continue

                    packed_name, comp_idx = PICDataReader._resolve_packed_component(path)
                    if packed_name not in packed_cache:
                        packed_var = f.inquire_variable(packed_name)
                        if packed_var is None:
                            raise KeyError
                        packed_cache[packed_name] = f.read(packed_var)
                    if packed_name == "all_1st_cc":
                        species_offsets = PICDataReader._infer_moment_species_offsets(
                            packed_cache[packed_name]
                        )
                        species, component_idx = comp_idx
                        results[path] = packed_cache[packed_name][
                            species_offsets[species] + component_idx, ...
                        ]
                    else:
                        results[path] = packed_cache[packed_name][comp_idx, ...]
                except Exception:
                    raise KeyError(f"Dataset '{path}' not found in file '{filename}'.")
        finally:
            f.close()
        return results

    @staticmethod
    def _resolve_packed_component(path: str):
        if path in PICDataReader.PACKED_FIELD_MAP:
            return PICDataReader.PACKED_FIELD_MAP[path]
        for idx, name in enumerate(PICDataReader._MOMENT_COMPONENTS):
            if path == f"{name}_i/p0/3d":
                return "all_1st_cc", ("i", idx)
            if path == f"{name}_e/p0/3d":
                return "all_1st_cc", ("e", idx)
        raise KeyError(path)

    @staticmethod
    def _infer_moment_species_offsets(data: np.ndarray) -> Dict[str, int]:
        """Infer electron/ion block ordering from the sign of the rho component."""
        rho0_mean = float(np.mean(data[0, ...]))
        rho1_mean = float(np.mean(data[13, ...]))
        if rho0_mean >= 0.0 and rho1_mean < 0.0:
            return {"i": 0, "e": 13}
        if rho1_mean >= 0.0 and rho0_mean < 0.0:
            return {"i": 13, "e": 0}
        # Fall back to the most common PSC convention if both are non-signed or ambiguous.
        return {"i": 13, "e": 0}

    @staticmethod
    def flatten_2d_slice(data_3d: np.ndarray) -> np.ndarray:
        """Flattens a 3D array representing a 2D slice into a 2D array."""
        if data_3d.ndim == 3 and data_3d.shape[2] == 1:
            return data_3d[:, :, 0]
        elif data_3d.ndim == 3 and data_3d.shape[0] == 1:
            return data_3d[0, :, :]
        return data_3d.squeeze()
