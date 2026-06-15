import h5py
import glob
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class PICDataReader:
    """Utility class to read and extract data from PIC simulation HDF5 output files."""
    
    @staticmethod
    def get_step_from_filename(filename: str) -> Optional[int]:
        """Extract the step from PSC field or particle output names."""
        match = re.search(r"\.(\d+)(?:_p\d+)?\.h5$", Path(filename).name)
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
                if step in file_map and file_map[step] != f:
                    raise ValueError(
                        f"Multiple HDF5 files found for step {step}: "
                        f"{file_map[step]} and {f}. This reader expects one "
                        "assembled p000000 file per snapshot."
                    )
                file_map[step] = f
        return file_map

    @staticmethod
    def discover_outputs(data_dir: str) -> Dict[str, object]:
        """Discover PSC outputs and reject ambiguous particle series."""
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {root}")

        fields = PICDataReader.find_files(str(root / "pfd.*_p*.h5"))
        moments = PICDataReader.find_files(str(root / "pfd_moments.*_p*.h5"))

        particle_series: Dict[str, Dict[int, str]] = {}
        for path in sorted(root.glob("prt*.h5")):
            step = PICDataReader.get_step_from_filename(str(path))
            match = re.match(r"(.+)\.\d+(?:_p\d+)?\.h5$", path.name)
            if step is None or match is None:
                continue
            particle_series.setdefault(match.group(1), {})[step] = str(path)

        return {
            "data_dir": root,
            "fields": fields,
            "moments": moments,
            "particles": particle_series,
        }

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
