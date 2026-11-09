"""Local field contrasts and complex coefficients in a periodic yz box (B0 along +z)."""
import numpy as np
from scipy.ndimage import gaussian_filter


def fundamental_transverse(snap):
    """Real, signed pair (n_z,n_y)=(±1,0) of both transverse components."""
    modes = []
    for axis in ("x", "y"):
        line = snap[axis].mean(axis=1)
        coeff = np.fft.rfft(line) / len(line)
        phase = 2 * np.pi * np.arange(len(line)) / len(line)
        modes.append((2 * (coeff[1] * np.exp(1j * phase)).real)[:, None])
    return np.hypot(*modes)


def attribution(snap, maps, catalog, sigma, b0):
    """Same filter and centre/exterior pair for all components. No branch labels."""
    bz = snap["z"]
    b = np.hypot(np.hypot(snap["x"], snap["y"]), bz)
    smooth = lambda x: gaussian_filter(x, sigma, mode="wrap")
    # b-bz remains defined even at a reversal. Subtract before filtering.
    wave = fundamental_transverse(snap)
    channels = {"B_over_B0": smooth(b) / b0, "Bz_over_B0": smooth(bz) / b0,
                "transverse_over_B0": smooth(b - bz) / b0,
                "fundamental_only_over_B0": np.broadcast_to(smooth(np.hypot(b0, wave) - b0) / b0, b.shape)}
    if "pperp_i" in snap:
        channels.update(n_i=smooth(snap["n_i"]), P_B=smooth(b*b/2),
                        Pperp_i=smooth(snap["pperp_i"]), Pperp_e=smooth(snap["pperp_e"]))
    occupied = maps["hole_mask"] | maps["peak_mask"]
    for obj in catalog:
        mask = maps[obj["kind"] + "_labels"] == obj["label"]
        # Adjacent exterior cells across all periodic faces.
        edge = np.zeros_like(mask)
        for axis in (0, 1):
            for shift in (-1, 1):
                edge |= np.roll(mask, shift, axis=axis)
        edge &= ~occupied
        score = maps["db"] if obj["kind"] == "hole" else -maps["db"]
        centre = np.unravel_index(np.argmin(np.where(mask, score, np.inf)), mask.shape)
        obj["exterior_cells"] = int(edge.sum())
        for name, values in channels.items():
            obj["contrast_" + name] = float(values[centre] - values[edge].mean()) if edge.any() else np.nan
        if "contrast_P_B" in obj:
            obj["contrast_P_total"] = sum(obj["contrast_" + k] for k in ("P_B", "Pperp_i", "Pperp_e"))
    return channels


def mode_coefficients(channels, nmax):
    """Complex spatial amplitudes, both oblique orientations, canonical conjugate pair."""
    nz, ny = channels["Bz_over_B0"].shape
    spectra = {key: np.fft.rfft2(value, s=(nz, ny), axes=(0, 1)) / (nz*ny)
               for key, value in channels.items() if key != "fundamental_only_over_B0"}
    rows = []
    for iy in range(min(nmax, (ny-1)//2) + 1):
        limit = min(nmax, (nz-1)//2)
        for iz in range(-limit if iy else 1, limit+1):
            row = {"n_parallel": iz, "n_perp_signed": iy}
            for key, spec in spectra.items():
                row[key + "_re"], row[key + "_im"] = spec[iz, iy].real, spec[iz, iy].imag
            rows.append(row)
    return rows
