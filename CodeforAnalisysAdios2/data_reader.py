import adios2
import glob
import re
import numpy as np
from typing import Dict, List, Optional

class PICDataReader:
    """Utility class to read and extract data from PIC simulation ADIOS2 output files."""

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
            for path in dataset_paths:
                try:
                    variable = f.inquire_variable(path)
                    if variable is None:
                        raise KeyError
                    results[path] = f.read(variable)
                except Exception:
                    raise KeyError(f"Dataset '{path}' not found in file '{filename}'.")
        finally:
            f.close()
        return results

    @staticmethod
    def flatten_2d_slice(data_3d: np.ndarray) -> np.ndarray:
        """Flattens a 3D array representing a 2D slice into a 2D array."""
        if data_3d.ndim == 3 and data_3d.shape[2] == 1:
            return data_3d[:, :, 0]
        elif data_3d.ndim == 3 and data_3d.shape[0] == 1:
            return data_3d[0, :, :]
        return data_3d.squeeze()
