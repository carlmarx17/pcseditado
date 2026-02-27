import h5py
import glob
import re
import numpy as np
from typing import Dict, List, Optional

class PICDataReader:
    """Utility class to read and extract data from PIC simulation HDF5 output files."""
    
    @staticmethod
    def get_step_from_filename(filename: str) -> Optional[int]:
        """Extracts the step number from a padded string like pfd_moments.001000_..."""
        match = re.search(r"\.(\d+)_", filename)
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
    def get_uid_group(f: h5py.File, base_prefix: str) -> str:
        """Finds the group with the given prefix in the HDF5 file."""
        for k in f.keys():
            if k.startswith(base_prefix):
                return k
        raise KeyError(f"No group starting with '{base_prefix}' found in file.")

    @staticmethod
    def read_multiple_fields_3d(filename: str, group_prefix: str, dataset_paths: List[str]) -> Dict[str, np.ndarray]:
        """Reads multiple 3D datasets from the same file and group to avoid re-opening."""
        results = {}
        with h5py.File(filename, 'r') as f:
            group_name = PICDataReader.get_uid_group(f, group_prefix)
            for path in dataset_paths:
                full_path = f"{group_name}/{path}"
                if full_path in f:
                    results[path] = f[full_path][()]
                else:
                    raise KeyError(f"Dataset '{path}' not found in group '{group_name}' of file '{filename}'")
        return results

    @staticmethod
    def flatten_2d_slice(data_3d: np.ndarray) -> np.ndarray:
        """Flattens a 3D array representing a 2D slice into a 2D array."""
        if data_3d.ndim == 3 and data_3d.shape[2] == 1:
            return data_3d[:, :, 0]
        elif data_3d.ndim == 3 and data_3d.shape[0] == 1:
            return data_3d[0, :, :]
        return data_3d.squeeze()
