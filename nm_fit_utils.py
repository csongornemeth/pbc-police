# nm_fit_utils.py

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_displacement(
    ref_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute displacement from reference to target.

    Parameters
    ----------
    ref_xyz
        Reference coordinates, shape (n_atoms, 3).
    target_xyz
        Target coordinates, shape (n_atoms, 3).

    Returns
    -------
    dx_atoms
        Per-atom displacement vectors, shape (n_atoms, 3).
    dx_flat
        Flattened displacement vector, shape (3*n_atoms,).
    disp_mag
        Per-atom displacement magnitudes, shape (n_atoms,).
    """
    ref_xyz = np.asarray(ref_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)

    if ref_xyz.shape != target_xyz.shape:
        raise ValueError(
            f"ref_xyz and target_xyz must have same shape. "
            f"Got {ref_xyz.shape} and {target_xyz.shape}"
        )

    dx_atoms = target_xyz - ref_xyz
    dx_flat = dx_atoms.reshape(-1)
    disp_mag = np.linalg.norm(dx_atoms, axis=1)

    return dx_atoms, dx_flat, disp_mag


def select_top_displacing_atoms(
    disp_mag: np.ndarray,
    n_top: int | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """
    Select atoms with largest displacement magnitudes.

    Use either n_top or percentile.

    Examples
    --------
    n_top=50
        Select 50 most displaced atoms.

    percentile=90
        Select atoms with displacement >= 90th percentile.
    """
    disp_mag = np.asarray(disp_mag, dtype=float)

    if n_top is None and percentile is None:
        raise ValueError("Provide either n_top or percentile.")

    if n_top is not None and percentile is not None:
        raise ValueError("Use only one of n_top or percentile.")

    if n_top is not None:
        n_top = int(n_top)

        if n_top <= 0:
            raise ValueError("n_top must be positive.")

        if n_top > disp_mag.size:
            raise ValueError(
                f"n_top={n_top} is larger than number of atoms={disp_mag.size}"
            )

        return np.argsort(disp_mag)[-n_top:]

    cutoff = np.percentile(disp_mag, percentile)
    return np.where(disp_mag >= cutoff)[0]


def modes_flat_to_atoms(
    modes_flat: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """
    Convert modes from shape (n_modes, 3N) to (n_modes, N, 3).
    """
    modes_flat = np.asarray(modes_flat, dtype=float)

    if modes_flat.ndim == 3:
        if modes_flat.shape[1] != n_atoms or modes_flat.shape[2] != 3:
            raise ValueError(
                f"3D modes should have shape (n_modes, {n_atoms}, 3). "
                f"Got {modes_flat.shape}"
            )
        return modes_flat

    if modes_flat.ndim != 2:
        raise ValueError(
            f"modes must be 2D or 3D. Got shape {modes_flat.shape}"
        )

    expected_dof = 3 * n_atoms

    if modes_flat.shape[1] != expected_dof:
        raise ValueError(
            f"Mode DOF mismatch. Got modes shape {modes_flat.shape}; "
            f"expected second dimension {expected_dof} for {n_atoms} atoms."
        )

    return modes_flat.reshape(modes_flat.shape[0], n_atoms, 3)


def slice_displacement_and_modes(
    dx_atoms: np.ndarray,
    modes: np.ndarray,
    atom_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice displacement and normal modes to selected atoms.

    Parameters
    ----------
    dx_atoms
        Shape (n_atoms, 3).
    modes
        Shape (n_modes, n_atoms, 3) or (n_modes, 3*n_atoms).
    atom_idx
        Local atom indices in the same basis as dx_atoms and modes.

    Returns
    -------
    dx_sel
        Shape (3*n_selected_atoms,).
    modes_sel
        Shape (n_modes, 3*n_selected_atoms).
    """
    dx_atoms = np.asarray(dx_atoms, dtype=float)
    atom_idx = np.asarray(atom_idx, dtype=int)

    if dx_atoms.ndim != 2 or dx_atoms.shape[1] != 3:
        raise ValueError(
            f"dx_atoms must have shape (n_atoms, 3). Got {dx_atoms.shape}"
        )

    n_atoms = dx_atoms.shape[0]
    modes_atoms = modes_flat_to_atoms(modes, n_atoms)

    dx_sel = dx_atoms[atom_idx].reshape(-1)
    modes_sel = modes_atoms[:, atom_idx, :].reshape(modes_atoms.shape[0], -1)

    return dx_sel, modes_sel


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return np.nan

    return float(np.dot(a, b) / denom)


def fit_single_modes(
    dx: np.ndarray,
    modes: np.ndarray,
    mode_number_offset: int = 7,
) -> pd.DataFrame:
    """
    Fit each normal mode individually.

    Model
    -----
    dx_fit_m = q_m * v_m

    with

    q_m = (dx · v_m) / (v_m · v_m)

    The denominator is kept because after slicing to top-displacing atoms,
    the sliced eigenvector is no longer guaranteed to have norm 1.
    """
    dx = np.asarray(dx, dtype=float)
    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2:
        raise ValueError(
            f"modes must have shape (n_modes, 3*n_selected_atoms). "
            f"Got {modes.shape}"
        )

    if modes.shape[1] != dx.size:
        raise ValueError(
            f"Mode vector length and displacement length do not match. "
            f"modes has {modes.shape[1]}, dx has {dx.size}"
        )

    rows = []
    dx_norm = np.linalg.norm(dx)

    for i, v in enumerate(modes):
        vv = float(np.dot(v, v))
        mode_norm = np.sqrt(vv)

        if vv == 0:
            q = np.nan
            dx_fit = np.zeros_like(dx)
        else:
            q = float(np.dot(dx, v) / vv)
            dx_fit = q * v

        fit_norm = np.linalg.norm(dx_fit)
        cos = cosine_similarity(dx, dx_fit)
        rms_error = float(np.sqrt(np.mean((dx - dx_fit) ** 2)))

        rows.append(
            {
                "mode_array_index": i,
                "bio3d_mode_number": i + mode_number_offset,
                "q_nm": q,
                "mode_norm_after_slicing": mode_norm,
                "dx_norm_nm": dx_norm,
                "fit_norm_nm": fit_norm,
                "fit_norm_fraction": fit_norm / dx_norm if dx_norm > 0 else np.nan,
                "cosine_overlap": cos,
                "abs_cosine_overlap": abs(cos) if not np.isnan(cos) else np.nan,
                "rms_error_nm": rms_error,
            }
        )

    return pd.DataFrame(rows)


def fit_mode_combination(
    dx: np.ndarray,
    modes: np.ndarray,
    mode_ids: np.ndarray | list[int] | range,
    mode_number_offset: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit a linear combination of selected normal modes.

    Model
    -----
    dx ≈ Vq

    where columns of V are selected mode vectors.

    Returns
    -------
    coeff_df
        Table of fitted q coefficients.
    summary
        Dictionary with fit metrics and reconstructed displacement.
    """
    dx = np.asarray(dx, dtype=float)
    modes = np.asarray(modes, dtype=float)
    mode_ids = np.asarray(list(mode_ids), dtype=int)

    if modes.ndim != 2:
        raise ValueError(
            f"modes must have shape (n_modes, 3*n_selected_atoms). "
            f"Got {modes.shape}"
        )

    if np.any(mode_ids < 0) or np.any(mode_ids >= modes.shape[0]):
        raise IndexError(
            f"mode_ids must be between 0 and {modes.shape[0] - 1}. "
            f"Got {mode_ids}"
        )

    V = modes[mode_ids].T

    coeffs, residuals, rank, singular_values = np.linalg.lstsq(
        V,
        dx,
        rcond=None,
    )

    dx_fit = V @ coeffs

    coeff_df = pd.DataFrame(
        {
            "mode_array_index": mode_ids,
            "bio3d_mode_number": mode_ids + mode_number_offset,
            "q_nm": coeffs,
        }
    )

    summary = {
        "n_modes_used": int(len(mode_ids)),
        "mode_array_start": int(mode_ids.min()) if len(mode_ids) else None,
        "mode_array_stop_inclusive": int(mode_ids.max()) if len(mode_ids) else None,
        "bio3d_mode_start": int(mode_ids.min() + mode_number_offset)
        if len(mode_ids)
        else None,
        "bio3d_mode_stop": int(mode_ids.max() + mode_number_offset)
        if len(mode_ids)
        else None,
        "cosine_overlap": cosine_similarity(dx, dx_fit),
        "rms_error_nm": float(np.sqrt(np.mean((dx - dx_fit) ** 2))),
        "dx_norm_nm": float(np.linalg.norm(dx)),
        "fit_norm_nm": float(np.linalg.norm(dx_fit)),
        "rank": int(rank),
        "residuals": residuals,
        "singular_values": singular_values,
        "dx_fit": dx_fit,
    }

    return coeff_df, summary