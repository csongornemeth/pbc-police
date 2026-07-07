#!/usr/bin/env python3
# nm_box_size_predictor_per_axis.py

"""
python nm_box_size_predictor_per_axis.py \
  --pdb 7lak \
  --target target_7lak.pdb \
  --reference target_7lak.pdb \
  --original-gro 7lak_prod.part0001.gro \
  --run-nma \
  --mode-start 6 \
  --n-modes 10 \
  --skip-existing-nma \
  --amplitudes 0 0.5 1 1.5 2 2.4 2.5 3 3.5 4 4.5 5 5.5 6.0 7.0 10.0 \
  --paddings 0.7 \
  --target-fit-selection "(protein and not element H)" \
  --plot-overlay \
  --save-all-trials \
  --save-best-helpers
  """

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from traj_utils import build_protein_heavy_views
from nma_bio3d_structure_modes import run_aanma_r_from_traj


# =============================================================================
# Basic utilities
# =============================================================================

def safe_float_label(value: float) -> str:
    value = float(value)

    if value.is_integer():
        label = str(int(value))
    else:
        label = f"{value:g}"

    return label.replace("-", "minus").replace(".", "p")


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    norm = np.linalg.norm(v)

    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector.")

    return v / norm


def load_single_frame(path: Path) -> md.Trajectory:
    traj = md.load(path.as_posix())

    if traj.n_frames != 1:
        traj = traj[0]

    return traj


def select_atoms(traj: md.Trajectory, selection: str) -> np.ndarray:
    idx = traj.topology.select(selection)

    if idx.size == 0:
        raise ValueError(f"Selection returned zero atoms: {selection}")

    return idx.astype(int)


def read_gro_box_lengths(gro_path: Path) -> np.ndarray:
    """
    Read the final line of a .gro file.

    For a rectangular box, the final line has:
        Lx Ly Lz

    For a triclinic GROMACS box, it can have 9 values.
    This prototype uses the first 3 values as the box lengths.
    """

    with open(gro_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty GRO file: {gro_path}")

    last = lines[-1].split()
    values = np.array([float(x) for x in last], dtype=float)

    if values.size < 3:
        raise ValueError(
            f"Could not read at least 3 box values from last line of {gro_path}: {lines[-1]}"
        )

    return values[:3]


def make_box_vectors_from_lengths(lengths: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = np.asarray(lengths, dtype=float)

    return np.array(
        [
            [Lx, 0.0, 0.0],
            [0.0, Ly, 0.0],
            [0.0, 0.0, Lz],
        ],
        dtype=float,
    )


def set_unitcell_vectors(traj: md.Trajectory, box_vectors: np.ndarray) -> md.Trajectory:
    out = traj[:]
    out.unitcell_vectors = box_vectors[None, :, :]
    return out


# =============================================================================
# Atom matching logic from bob_the_boxer.py
# =============================================================================

def atom_base_key(atom):
    """
    Non-unique biological atom identity.

    This is the original bob_the_boxer-style key, but it can be duplicated
    when residue numbers repeat inside the same chain.
    """
    res = atom.residue
    chain = res.chain

    return (
        chain.index,
        res.resSeq,
        res.name,
        atom.name,
    )


def atom_key(atom):
    """
    Backwards-compatible alias for diagnostics or older calls.

    Do not use this alone as a dictionary key when uniqueness is required;
    build_atom_key_map() adds an occurrence counter for robustness.
    """
    return atom_base_key(atom)


def build_atom_key_map(traj: md.Trajectory, selection: str) -> dict:
    """
    Build a robust atom map for matching structures.

    The old key was:
        (chain.index, resSeq, resname, atomname)

    That can fail when residue numbers repeat. The new key is:
        (chain.index, resSeq, resname, atomname, occurrence)

    The occurrence counter preserves the selected atom order while making
    repeated residue/atom labels unique.
    """
    indices = traj.topology.select(selection)

    if indices.size == 0:
        raise ValueError(f"Selection returned zero atoms: {selection}")

    out = {}
    counts = {}

    for idx in indices:
        atom = traj.topology.atom(int(idx))
        base_key = atom_base_key(atom)

        occurrence = counts.get(base_key, 0)
        counts[base_key] = occurrence + 1

        key = base_key + (occurrence,)

        if key in out:
            raise ValueError(
                f"Internal duplicate atom key found even after occurrence indexing: {key}"
            )

        out[key] = int(idx)

    return out


def common_atom_indices_by_key(
    reference: md.Trajectory,
    target: md.Trajectory,
    helpers: list[md.Trajectory],
    selection: str,
):
    ref_map = build_atom_key_map(reference, selection)
    target_map = build_atom_key_map(target, selection)
    helper_maps = [build_atom_key_map(h, selection) for h in helpers]

    common_keys = set(ref_map.keys()) & set(target_map.keys())

    for hm in helper_maps:
        common_keys &= set(hm.keys())

    if not common_keys:
        raise ValueError(f"No common atoms found for selection: {selection}")

    common_keys = sorted(common_keys)

    ref_idx = np.array([ref_map[k] for k in common_keys], dtype=int)
    target_idx = np.array([target_map[k] for k in common_keys], dtype=int)

    helper_indices = [
        np.array([hm[k] for k in common_keys], dtype=int)
        for hm in helper_maps
    ]

    return ref_idx, target_idx, helper_indices, common_keys


def align_to_reference_mapped(
    target: md.Trajectory,
    helpers: list[md.Trajectory],
    reference: md.Trajectory,
    ref_align_indices: np.ndarray,
    target_align_indices: np.ndarray,
    helper_align_indices: list[np.ndarray],
) -> tuple[md.Trajectory, list[md.Trajectory]]:

    target_aligned = target[:]
    target_aligned.superpose(
        reference,
        frame=0,
        atom_indices=target_align_indices,
        ref_atom_indices=ref_align_indices,
    )

    helpers_aligned = []

    for helper, h_idx in zip(helpers, helper_align_indices):
        h = helper[:]
        h.superpose(
            reference,
            frame=0,
            atom_indices=h_idx,
            ref_atom_indices=ref_align_indices,
        )
        helpers_aligned.append(h)

    return target_aligned, helpers_aligned


# =============================================================================
# COM centring
# =============================================================================

def atom_masses(topology: md.Topology, atom_indices: np.ndarray) -> np.ndarray:
    masses = []

    for idx in atom_indices:
        atom = topology.atom(int(idx))

        if atom.element is None or atom.element.mass is None:
            masses.append(12.0)
        else:
            masses.append(float(atom.element.mass))

    return np.array(masses, dtype=float)


def compute_com(
    xyz: np.ndarray,
    atom_indices: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    coords = xyz[atom_indices]
    return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)


def centre_by_com_single(
    traj: md.Trajectory,
    atom_indices: np.ndarray,
) -> md.Trajectory:
    out = traj[:]
    masses = atom_masses(out.topology, atom_indices)
    com = compute_com(out.xyz[0], atom_indices, masses)
    out.xyz[0] -= com

    return out


# =============================================================================
# Oriented box fitting logic from bob_the_boxer.py
# =============================================================================

def pca_axes(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    X = points - center

    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T

    if np.linalg.det(axes) < 0:
        axes[2] *= -1.0

    return axes


def oriented_box_from_points(
    points: np.ndarray,
    axes: np.ndarray,
    padding: float,
) -> dict:
    projected = points @ axes.T

    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    extents = maxs - mins
    lengths = extents + 2.0 * float(padding)

    box_center_projected = 0.5 * (mins + maxs)

    return {
        "axes": axes,
        "mins_projected": mins,
        "maxs_projected": maxs,
        "extents": extents,
        "lengths": lengths,
        "box_center_projected": box_center_projected,
    }


def transform_to_box_frame(
    traj: md.Trajectory,
    axes: np.ndarray,
    box_center_projected: np.ndarray,
    lengths: np.ndarray,
) -> md.Trajectory:
    out = traj[:]

    projected = out.xyz[0] @ axes.T
    projected = projected - box_center_projected[None, :]
    projected = projected + 0.5 * lengths[None, :]

    out.xyz[0] = projected

    return out


def fit_common_box(
    target: md.Trajectory,
    helpers: list[md.Trajectory],
    reference: md.Trajectory,
    common_align_selection: str,
    target_fit_selection: str,
    helper_fit_selection: str,
    center_selection: str,
    padding: float,
    fixed_axes: np.ndarray | None = None,
) -> dict:

    (
        ref_align_idx,
        target_align_idx,
        helper_align_idx,
        common_keys,
    ) = common_atom_indices_by_key(
        reference=reference,
        target=target,
        helpers=helpers,
        selection=common_align_selection,
    )

    target_fit_indices = select_atoms(target, target_fit_selection)

    helper_fit_indices = [
        select_atoms(h, helper_fit_selection)
        for h in helpers
    ]

    target_center_indices = select_atoms(target, center_selection)

    helper_center_indices = [
        select_atoms(h, center_selection)
        for h in helpers
    ]

    target_aligned, helpers_aligned = align_to_reference_mapped(
        target=target,
        helpers=helpers,
        reference=reference,
        ref_align_indices=ref_align_idx,
        target_align_indices=target_align_idx,
        helper_align_indices=helper_align_idx,
    )

    target_centred = centre_by_com_single(
        traj=target_aligned,
        atom_indices=target_center_indices,
    )

    helpers_centred = [
        centre_by_com_single(h, idx)
        for h, idx in zip(helpers_aligned, helper_center_indices)
    ]

    helper_fit_points = [
        h.xyz[0, fit_idx, :]
        for h, fit_idx in zip(helpers_centred, helper_fit_indices)
    ]

    point_cloud = np.concatenate(
        [target_centred.xyz[0, target_fit_indices, :]]
        + helper_fit_points,
        axis=0,
    )

    if fixed_axes is None:
        axes = pca_axes(point_cloud)
    else:
        axes = np.asarray(fixed_axes, dtype=float)
        if axes.shape != (3, 3):
            raise ValueError(f"fixed_axes must have shape (3, 3), got {axes.shape}")

    box_info = oriented_box_from_points(
        points=point_cloud,
        axes=axes,
        padding=padding,
    )

    lengths = box_info["lengths"]
    box_center_projected = box_info["box_center_projected"]
    box_vectors = make_box_vectors_from_lengths(lengths)

    target_box_frame = transform_to_box_frame(
        traj=target_centred,
        axes=axes,
        box_center_projected=box_center_projected,
        lengths=lengths,
    )

    helpers_box_frame = [
        transform_to_box_frame(
            traj=h,
            axes=axes,
            box_center_projected=box_center_projected,
            lengths=lengths,
        )
        for h in helpers_centred
    ]

    target_boxed = set_unitcell_vectors(target_box_frame, box_vectors)

    helpers_boxed = [
        set_unitcell_vectors(h, box_vectors)
        for h in helpers_box_frame
    ]

    return {
        "target_boxed": target_boxed,
        "helpers_boxed": helpers_boxed,
        "target_fit_indices": target_fit_indices,
        "helper_fit_indices": helper_fit_indices,
        "common_align_atoms": len(common_keys),
        "point_cloud": point_cloud,
        "axes": axes,
        "lengths": lengths,
        "box_vectors": box_vectors,
        "box_info": box_info,
    }


# =============================================================================
# NM helper generation
# =============================================================================

def make_nm_helpers(
    helper_reference: md.Trajectory,
    nma_modes: np.ndarray,
    mode_start: int,
    n_modes: int,
    amplitude: float,
) -> tuple[list[md.Trajectory], list[dict]]:
    """
    Generate plus/minus helper structures for individual normal modes.

    Critical:
        helper_reference.xyz[0] must have the SAME atom order as raw_modes_all.npy.

    This mirrors the safe logic from your older save_endpoint_structures():
        - use xyz0 from the reference trajectory
        - reshape the 3N vector to (N, 3)
        - create fresh md.Trajectory objects with the same topology
    """

    if helper_reference.n_frames < 1:
        raise ValueError("helper_reference must contain at least one frame.")

    xyz0 = np.asarray(helper_reference.xyz[0], dtype=float)
    n_atoms = helper_reference.n_atoms
    expected_dof = n_atoms * 3

    nma_modes = np.asarray(nma_modes, dtype=float)

    if nma_modes.ndim != 2:
        raise ValueError(f"Expected nma_modes to be 2D, got {nma_modes.shape}")

    if nma_modes.shape[0] != expected_dof:
        raise ValueError(
            "NMA mode matrix does not match helper reference atom count:\n"
            f"  helper atoms:      {n_atoms}\n"
            f"  expected DOF:      {expected_dof}\n"
            f"  nma_modes rows:    {nma_modes.shape[0]}\n"
            "This means raw_modes_all.npy was generated from a different atom set/order."
        )

    mode_stop = mode_start + n_modes

    if mode_start < 0:
        raise ValueError("--mode-start must be >= 0")

    if mode_stop > nma_modes.shape[1]:
        raise ValueError(
            f"Requested modes [{mode_start}:{mode_stop}], "
            f"but nma_modes only has {nma_modes.shape[1]} columns."
        )

    helpers = []
    metadata = []

    for mode_index in range(mode_start, mode_stop):
        raw_mode = nma_modes[:, mode_index]
        dr_dir = normalize_vector(raw_mode)
        dr_scaled_atoms = (float(amplitude) * dr_dir).reshape(n_atoms, 3)

        xyz_minus = xyz0 - dr_scaled_atoms
        xyz_plus = xyz0 + dr_scaled_atoms

        traj_minus = md.Trajectory(
            xyz=xyz_minus[None, :, :],
            topology=helper_reference.topology,
        )

        traj_plus = md.Trajectory(
            xyz=xyz_plus[None, :, :],
            topology=helper_reference.topology,
        )

        bio3d_mode_number = mode_index + 1

        helpers.append(traj_minus)
        metadata.append(
            {
                "mode_index_zero_based": int(mode_index),
                "bio3d_mode_number": int(bio3d_mode_number),
                "sign": "minus",
                "amplitude_nm": float(amplitude),
            }
        )

        helpers.append(traj_plus)
        metadata.append(
            {
                "mode_index_zero_based": int(mode_index),
                "bio3d_mode_number": int(bio3d_mode_number),
                "sign": "plus",
                "amplitude_nm": float(amplitude),
            }
        )

    return helpers, metadata


def save_helpers(
    helpers: list[md.Trajectory],
    metadata: list[dict],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for h, meta in zip(helpers, metadata):
        amp_label = safe_float_label(meta["amplitude_nm"])
        mode_number = int(meta["bio3d_mode_number"])
        sign = meta["sign"]

        path = outdir / f"mode{mode_number:03d}_{sign}_amp{amp_label}nm.pdb"
        h.save_pdb(path.as_posix())


# =============================================================================
# Scoring
# =============================================================================

def score_box(
    predicted_lengths: np.ndarray,
    target_lengths: np.ndarray,
    volume_weight: float,
    undersize_weight: float = 1000.0,
    undersize_tolerance: float = 0.0,
) -> dict:
    """
    Score a candidate box by comparing x/y/z separately.

    Important difference from the old version:
        - no sorting of box lengths
        - each direction is compared directly: Lx to Lx, Ly to Ly, Lz to Lz
        - any side shorter than the target gets a strong penalty

    Positive error  = predicted side is longer than target side.
    Negative error  = predicted side is shorter than target side.
    Shortfall       = max(target - predicted, 0).
    """

    predicted_lengths = np.asarray(predicted_lengths, dtype=float)
    target_lengths = np.asarray(target_lengths, dtype=float)

    if predicted_lengths.shape != (3,):
        raise ValueError(f"predicted_lengths must have shape (3,), got {predicted_lengths.shape}")

    if target_lengths.shape != (3,):
        raise ValueError(f"target_lengths must have shape (3,), got {target_lengths.shape}")

    length_errors = predicted_lengths - target_lengths
    side_shortfalls = np.maximum(target_lengths - predicted_lengths, 0.0)

    length_rmse = float(np.sqrt(np.mean(length_errors ** 2)))
    max_shortfall = float(np.max(side_shortfalls))
    total_shortfall = float(np.sum(side_shortfalls))
    is_undersized = bool(np.any(side_shortfalls > float(undersize_tolerance)))

    predicted_volume = float(np.prod(predicted_lengths))
    target_volume = float(np.prod(target_lengths))

    volume_rel_error = float(abs(predicted_volume - target_volume) / target_volume)
    undersize_penalty = float(float(undersize_weight) * total_shortfall)

    score = float(
        length_rmse
        + float(volume_weight) * volume_rel_error
        + undersize_penalty
    )

    return {
        "score": score,
        "length_rmse_nm": length_rmse,
        "volume_rel_error": volume_rel_error,
        "undersize_penalty": undersize_penalty,
        "max_shortfall_nm": max_shortfall,
        "total_shortfall_nm": total_shortfall,
        "is_undersized": is_undersized,
        "x_shortfall_nm": float(side_shortfalls[0]),
        "y_shortfall_nm": float(side_shortfalls[1]),
        "z_shortfall_nm": float(side_shortfalls[2]),
        "x_error_nm": float(length_errors[0]),
        "y_error_nm": float(length_errors[1]),
        "z_error_nm": float(length_errors[2]),
        "predicted_volume_nm3": predicted_volume,
        "target_volume_nm3": target_volume,
        "predicted_lengths_nm": predicted_lengths.tolist(),
        "target_lengths_nm": target_lengths.tolist(),
    }


def select_best_per_axis(
    rows: list[dict],
    target_box_lengths: np.ndarray,
    undersize_tolerance: float = 0.0,
) -> dict:
    """
    Select the best one-mode/amplitude candidate separately for x, y, and z.

    A candidate is acceptable for an axis only if that side is not shorter than
    the target side by more than undersize_tolerance.

    Among acceptable candidates, the best candidate for an axis is the one with
    the smallest positive/absolute error for that same axis.
    """

    target_box_lengths = np.asarray(target_box_lengths, dtype=float)

    axis_specs = [
        ("x", "pred_Lx_nm", "target_Lx_nm", 0),
        ("y", "pred_Ly_nm", "target_Ly_nm", 1),
        ("z", "pred_Lz_nm", "target_Lz_nm", 2),
    ]

    selected = {}

    for axis_name, pred_key, target_key, _axis_i in axis_specs:
        acceptable = []

        for row in rows:
            pred = float(row[pred_key])
            target = float(row[target_key])
            error = pred - target

            if error < -float(undersize_tolerance):
                continue

            candidate = dict(row)
            candidate[f"{axis_name}_axis_error_nm"] = error
            candidate[f"{axis_name}_axis_abs_error_nm"] = abs(error)
            acceptable.append(candidate)

        if not acceptable:
            raise RuntimeError(
                f"No acceptable candidate found for {axis_name}-axis. "
                "Try larger amplitudes and/or larger padding."
            )

        selected[axis_name] = min(
            acceptable,
            key=lambda r: r[f"{axis_name}_axis_abs_error_nm"],
        )

    final_lengths = np.array(
        [
            float(selected["x"]["pred_Lx_nm"]),
            float(selected["y"]["pred_Ly_nm"]),
            float(selected["z"]["pred_Lz_nm"]),
        ],
        dtype=float,
    )

    # Absolute safety: the final assembled box is never allowed to be shorter
    # than the original/reference GRO box in any direction.
    final_lengths = np.maximum(final_lengths, target_box_lengths)

    return {
        "selected_x": selected["x"],
        "selected_y": selected["y"],
        "selected_z": selected["z"],
        "final_lengths_nm": final_lengths.tolist(),
        "target_lengths_nm": target_box_lengths.tolist(),
        "final_volume_nm3": float(np.prod(final_lengths)),
        "target_volume_nm3": float(np.prod(target_box_lengths)),
        "final_minus_target_nm": (final_lengths - target_box_lengths).tolist(),
    }


def transform_target_to_final_box(
    target: md.Trajectory,
    reference: md.Trajectory,
    common_align_selection: str,
    target_fit_selection: str,
    center_selection: str,
    axes: np.ndarray,
    lengths: np.ndarray,
) -> dict:
    """
    Align target to reference, centre it by COM, rotate into a fixed box frame,
    centre it inside the final per-axis box, and attach orthorhombic box vectors.
    """

    axes = np.asarray(axes, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    if axes.shape != (3, 3):
        raise ValueError(f"axes must have shape (3, 3), got {axes.shape}")

    if lengths.shape != (3,):
        raise ValueError(f"lengths must have shape (3,), got {lengths.shape}")

    ref_map = build_atom_key_map(reference, common_align_selection)
    target_map = build_atom_key_map(target, common_align_selection)

    common_keys = sorted(set(ref_map.keys()) & set(target_map.keys()))

    if not common_keys:
        raise ValueError(f"No common atoms found for selection: {common_align_selection}")

    ref_align_idx = np.array([ref_map[k] for k in common_keys], dtype=int)
    target_align_idx = np.array([target_map[k] for k in common_keys], dtype=int)

    target_aligned = target[:]
    target_aligned.superpose(
        reference,
        frame=0,
        atom_indices=target_align_idx,
        ref_atom_indices=ref_align_idx,
    )

    target_center_indices = select_atoms(target_aligned, center_selection)
    target_centred = centre_by_com_single(target_aligned, target_center_indices)

    target_fit_indices = select_atoms(target_centred, target_fit_selection)
    target_fit_points = target_centred.xyz[0, target_fit_indices, :]
    projected_fit = target_fit_points @ axes.T

    mins = projected_fit.min(axis=0)
    maxs = projected_fit.max(axis=0)
    box_center_projected = 0.5 * (mins + maxs)

    target_box_frame = transform_to_box_frame(
        traj=target_centred,
        axes=axes,
        box_center_projected=box_center_projected,
        lengths=lengths,
    )

    box_vectors = make_box_vectors_from_lengths(lengths)
    target_boxed = set_unitcell_vectors(target_box_frame, box_vectors)

    return {
        "target_boxed": target_boxed,
        "target_fit_indices": target_fit_indices,
        "box_vectors": box_vectors,
        "box_center_projected": box_center_projected,
        "common_align_atoms": len(common_keys),
    }

# =============================================================================
# Plotting
# =============================================================================

def box_corners_from_lengths(lengths: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = lengths

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [Lx, 0.0, 0.0],
            [Lx, Ly, 0.0],
            [0.0, Ly, 0.0],
            [0.0, 0.0, Lz],
            [Lx, 0.0, Lz],
            [Lx, Ly, Lz],
            [0.0, Ly, Lz],
        ],
        dtype=float,
    )


def box_edges_from_corners():
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def set_equal_axes_3d(ax, lengths: np.ndarray) -> None:
    Lx, Ly, Lz = lengths
    radius = 0.5 * max(Lx, Ly, Lz)

    cx = 0.5 * Lx
    cy = 0.5 * Ly
    cz = 0.5 * Lz

    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(cz - radius, cz + radius)


def plot_overlay_and_box_3d(
    trajs_box_frame: list[md.Trajectory],
    fit_indices_per_traj: list[np.ndarray],
    lengths: np.ndarray,
    out_png: Path,
    max_points_per_structure: int = 5000,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, tr in enumerate(trajs_box_frame):
        fit_idx = fit_indices_per_traj[i]
        pts = tr.xyz[0, fit_idx, :]

        if pts.shape[0] > max_points_per_structure:
            sel = np.linspace(
                0,
                pts.shape[0] - 1,
                max_points_per_structure,
            ).astype(int)
            pts = pts[sel]

        if i == 0:
            label = "target"
            alpha = 0.9
            size = 6
        else:
            label = f"helper_{i}"
            alpha = 0.25
            size = 3

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=size,
            alpha=alpha,
            label=label,
        )

    corners = box_corners_from_lengths(lengths)

    for i, j in box_edges_from_corners():
        p1 = corners[i]
        p2 = corners[j]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            linewidth=1.5,
        )

    ax.set_xlabel("x / nm")
    ax.set_ylabel("y / nm")
    ax.set_zlabel("z / nm")
    ax.set_title("Target + NM helpers in common box frame")
    ax.legend(loc="best", fontsize=8)

    set_equal_axes_3d(ax, lengths)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_overlay_projection(
    trajs_box_frame: list[md.Trajectory],
    fit_indices_per_traj: list[np.ndarray],
    lengths: np.ndarray,
    out_png: Path,
    plane: str,
    max_points_per_structure: int = 5000,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plane = plane.lower()

    plane_map = {
        "xy": (0, 1, lengths[0], lengths[1], "x / nm", "y / nm"),
        "xz": (0, 2, lengths[0], lengths[2], "x / nm", "z / nm"),
        "yz": (1, 2, lengths[1], lengths[2], "y / nm", "z / nm"),
    }

    if plane not in plane_map:
        raise ValueError("plane must be one of: xy, xz, yz")

    a, b, La, Lb, xlabel, ylabel = plane_map[plane]

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, tr in enumerate(trajs_box_frame):
        fit_idx = fit_indices_per_traj[i]
        pts = tr.xyz[0, fit_idx, :]

        if pts.shape[0] > max_points_per_structure:
            sel = np.linspace(
                0,
                pts.shape[0] - 1,
                max_points_per_structure,
            ).astype(int)
            pts = pts[sel]

        if i == 0:
            label = "target"
            alpha = 0.9
            size = 6
        else:
            label = f"helper_{i}"
            alpha = 0.25
            size = 3

        ax.scatter(
            pts[:, a],
            pts[:, b],
            s=size,
            alpha=alpha,
            label=label,
        )

    ax.plot(
        [0.0, La, La, 0.0, 0.0],
        [0.0, 0.0, Lb, Lb, 0.0],
        linewidth=1.5,
    )

    ax.set_xlim(0.0, La)
    ax.set_ylim(0.0, Lb)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Overlay projection: {plane.upper()}")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def write_sweep_results_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "trial",
        "bio3d_mode_number",
        "mode_index_zero_based",
        "amplitude_nm",
        "padding_nm",
        "score",
        "length_rmse_nm",
        "volume_rel_error",
        "undersize_penalty",
        "max_shortfall_nm",
        "total_shortfall_nm",
        "is_undersized",
        "x_shortfall_nm",
        "y_shortfall_nm",
        "z_shortfall_nm",
        "x_error_nm",
        "y_error_nm",
        "z_error_nm",
        "predicted_volume_nm3",
        "target_volume_nm3",
        "pred_Lx_nm",
        "pred_Ly_nm",
        "pred_Lz_nm",
        "target_Lx_nm",
        "target_Ly_nm",
        "target_Lz_nm",
        "n_helpers",
        "mode_start_zero_based",
        "n_modes",
        "first_bio3d_mode",
        "last_bio3d_mode",
        "common_align_atoms",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype NM-guided box size predictor. "
            "Generates individual plus/minus NM helper structures with uniform amplitude, "
            "fits a bob_the_boxer-style common oriented box, and compares it to an original GRO box."
        )
    )

    parser.add_argument("--pdb", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--original-gro", required=True, type=Path)

    parser.add_argument(
        "--outdir",
        default=None,
        type=Path,
        help="Default: boxpred_results/{pdb}",
    )

    parser.add_argument(
        "--nma-dir",
        default=None,
        type=Path,
        help="Default: boxpred_results/{pdb}/raw_data/nma",
    )

    parser.add_argument(
        "--run-nma",
        action="store_true",
        help="Run Bio3D NMA before loading raw_modes_all.npy.",
    )

    parser.add_argument(
        "--skip-existing-nma",
        action="store_true",
        help="If --run-nma is used and raw_modes_all.npy exists, skip rerunning NMA.",
    )

    parser.add_argument(
        "--nma-keep",
        type=int,
        default=1000,
        help="Number of NMA modes to keep when running NMA.",
    )

    parser.add_argument(
        "--mode-start",
        type=int,
        default=6,
        help="Zero-based mode start. Bio3D mode 7 corresponds to --mode-start 6.",
    )

    parser.add_argument(
        "--n-modes",
        type=int,
        default=5,
        help="Number of individual NMs to use.",
    )

    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        required=True,
        help="Uniform amplitudes in nm.",
    )

    parser.add_argument(
        "--paddings",
        type=float,
        nargs="+",
        required=True,
        help="One-sided paddings in nm. Final length = extent + 2 * padding.",
    )

    parser.add_argument(
        "--volume-weight",
        type=float,
        default=1.0,
        help="Weight for relative volume error in score.",
    )

    parser.add_argument(
        "--undersize-weight",
        type=float,
        default=1000.0,
        help="Penalty weight for total shortfall in nm. Larger = stricter rejection of short sides.",
    )

    parser.add_argument(
        "--undersize-tolerance",
        type=float,
        default=0.0,
        help="Allowed side-specific shortfall in nm before a candidate is considered undersized.",
    )

    parser.add_argument(
        "--common-align-selection",
        default="protein and backbone",
        help="Selection used to match common atoms between reference, target, and helpers.",
    )

    parser.add_argument(
        "--target-fit-selection",
        default="protein and not element H",
        help="Target atoms used for fitting. Include ligand here if needed.",
    )

    parser.add_argument(
        "--helper-fit-selection",
        default="protein and not element H",
        help="Helper atoms used for fitting.",
    )

    parser.add_argument(
        "--center-selection",
        default="protein and not element H",
        help="Atoms used for COM centring.",
    )

    parser.add_argument(
        "--save-best-helpers",
        action="store_true",
        help="Save plus/minus helper PDBs for the best amplitude.",
    )

    parser.add_argument(
        "--save-all-trials",
        action="store_true",
        help="Save boxed target structures and raw trial summaries for every amplitude/padding pair.",
    )

    parser.add_argument(
        "--plot-overlay",
        action="store_true",
        help="Save overlay plots for the best solution.",
    )

    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    pdb_code = args.pdb.strip().lower()

    if args.outdir is None:
        outdir = Path("boxpred_results") / pdb_code
    else:
        outdir = args.outdir

    structures_dir = outdir / "structures"
    raw_data_dir = outdir / "raw_data"
    plots_dir = outdir / "plots"

    if args.nma_dir is None:
        nma_dir = raw_data_dir / "nma"
    else:
        nma_dir = args.nma_dir

    structures_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    nma_dir.mkdir(parents=True, exist_ok=True)

    raw_modes_all_path = nma_dir / "raw_modes_all.npy"

    print("====================================")
    print("NM box size predictor prototype")
    print("====================================")
    print(f"PDB/system:       {pdb_code}")
    print(f"Target:           {args.target}")
    print(f"Reference:        {args.reference}")
    print(f"Original GRO:     {args.original_gro}")
    print(f"Output dir:       {outdir}")
    print(f"Structures dir:   {structures_dir}")
    print(f"Raw data dir:     {raw_data_dir}")
    print(f"Plots dir:        {plots_dir}")
    print(f"NMA dir:          {nma_dir}")
    print(f"raw_modes_all:    {raw_modes_all_path}")
    print(f"Mode start:       {args.mode_start}")
    print(f"N modes:          {args.n_modes}")
    print(f"Amplitudes:       {args.amplitudes}")
    print(f"Paddings:         {args.paddings}")
    print("")

    # -------------------------------------------------------------------------
    # 1. Read target/original MD box
    # -------------------------------------------------------------------------

    target_box_lengths = read_gro_box_lengths(args.original_gro)
    target_box_volume = float(np.prod(target_box_lengths))

    print("[Original GRO box]")
    print(f"  lengths nm: {target_box_lengths}")
    print(f"  volume nm3: {target_box_volume:.6f}")
    print("")

    # -------------------------------------------------------------------------
    # 2. Build/load the same protein-heavy reference used for NMA
    # -------------------------------------------------------------------------

    print("[Reference for NMA helpers]")
    print("  Using build_protein_heavy_views(pdb), same logic as older working code.")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    helper_reference = traj_protein_heavy_ref
    expected_dof = helper_reference.n_atoms * 3

    print(f"  helper reference atoms: {helper_reference.n_atoms}")
    print(f"  expected DOF:           {expected_dof}")
    print("")

    # -------------------------------------------------------------------------
    # 3. Optional NMA run
    # -------------------------------------------------------------------------

    if args.run_nma:
        if args.skip_existing_nma and raw_modes_all_path.exists():
            print("[NMA] Skipping NMA because raw_modes_all.npy exists.")
        else:
            print("[NMA] Running Bio3D aanma.pdb...")
            run_aanma_r_from_traj(
                traj_protein_heavy=helper_reference,
                n_modes_keep=args.nma_keep,
                save_raw_modes_dir=nma_dir,
                save_prefix=None,
            )

    if not raw_modes_all_path.exists():
        raise FileNotFoundError(
            f"Missing NMA mode file: {raw_modes_all_path}\n"
            "Either pass --nma-dir pointing to an existing raw_modes_all.npy, "
            "or use --run-nma."
        )

    nma_modes = np.load(raw_modes_all_path)

    print("[NMA modes]")
    print(f"  shape: {nma_modes.shape}")

    if nma_modes.ndim != 2:
        raise ValueError(f"Expected raw_modes_all.npy to be 2D, got {nma_modes.shape}")

    if nma_modes.shape[0] != expected_dof:
        raise ValueError(
            "raw_modes_all.npy does not match helper_reference atom order/count:\n"
            f"  raw_modes_all rows: {nma_modes.shape[0]}\n"
            f"  expected DOF:       {expected_dof}\n"
            "This means this NMA file was generated from a different atom set/order."
        )

    print("  NMA modes match helper reference DOF.")
    print("")

    # -------------------------------------------------------------------------
    # 4. Load target and alignment reference
    # -------------------------------------------------------------------------

    target = load_single_frame(args.target)
    reference = load_single_frame(args.reference)

    print("[Loaded structures]")
    print(f"  target atoms:    {target.n_atoms}")
    print(f"  reference atoms: {reference.n_atoms}")
    print("")

    # -------------------------------------------------------------------------
    # 5. Build one fixed PCA box frame for the whole sweep
    # -------------------------------------------------------------------------

    # This is important for the per-axis method.  If each one-mode trial were
    # allowed to define its own PCA axes, then Lx from one trial and Ly from
    # another trial would not necessarily refer to the same directions.
    # Therefore, we first define a fixed box frame from all selected modes at
    # the largest tested amplitude, and all later trials use this same frame.

    max_amplitude = float(max(args.amplitudes))

    print("[Fixed box frame]")
    print(
        "  Building fixed axes from all selected modes "
        f"at max amplitude {max_amplitude} nm."
    )

    fixed_helpers, fixed_helper_metadata = make_nm_helpers(
        helper_reference=helper_reference,
        nma_modes=nma_modes,
        mode_start=args.mode_start,
        n_modes=args.n_modes,
        amplitude=max_amplitude,
    )

    fixed_fit = fit_common_box(
        target=target,
        helpers=fixed_helpers,
        reference=reference,
        common_align_selection=args.common_align_selection,
        target_fit_selection=args.target_fit_selection,
        helper_fit_selection=args.helper_fit_selection,
        center_selection=args.center_selection,
        padding=0.0,
        fixed_axes=None,
    )

    fixed_axes = fixed_fit["axes"]
    np.save(raw_data_dir / "fixed_box_axes_rows.npy", fixed_axes)

    with open(raw_data_dir / "fixed_box_frame_summary.json", "w") as f:
        json.dump(
            {
                "method": "PCA axes from target plus all selected NM helpers at max amplitude",
                "max_amplitude_nm": max_amplitude,
                "mode_start_zero_based": int(args.mode_start),
                "n_modes": int(args.n_modes),
                "first_bio3d_mode": int(args.mode_start + 1),
                "last_bio3d_mode": int(args.mode_start + args.n_modes),
                "axes_rows": fixed_axes.tolist(),
                "helper_metadata": fixed_helper_metadata,
            },
            f,
            indent=2,
        )

    print("  fixed axes rows:")
    print(fixed_axes)
    print("")

    # -------------------------------------------------------------------------
    # 6. Per-mode/per-amplitude sweep using the fixed box frame
    # -------------------------------------------------------------------------

    rows = []
    trial_records = []
    best_global = None

    trial_counter = 0
    mode_stop = args.mode_start + args.n_modes

    for mode_index in range(args.mode_start, mode_stop):
        bio3d_mode_number = mode_index + 1

        print(f"[Mode] Bio3D mode {bio3d_mode_number} / zero-based {mode_index}")

        for amplitude in args.amplitudes:
            print(f"  [Amplitude] {amplitude} nm")

            # One trial now contains only one NM, plus and minus.
            # This lets each axis later choose its own best NM/amplitude.
            helpers, helper_metadata = make_nm_helpers(
                helper_reference=helper_reference,
                nma_modes=nma_modes,
                mode_start=mode_index,
                n_modes=1,
                amplitude=amplitude,
            )

            for padding in args.paddings:
                trial_counter += 1

                fit = fit_common_box(
                    target=target,
                    helpers=helpers,
                    reference=reference,
                    common_align_selection=args.common_align_selection,
                    target_fit_selection=args.target_fit_selection,
                    helper_fit_selection=args.helper_fit_selection,
                    center_selection=args.center_selection,
                    padding=padding,
                    fixed_axes=fixed_axes,
                )

                predicted_lengths = fit["lengths"]

                score_info = score_box(
                    predicted_lengths=predicted_lengths,
                    target_lengths=target_box_lengths,
                    volume_weight=args.volume_weight,
                    undersize_weight=args.undersize_weight,
                    undersize_tolerance=args.undersize_tolerance,
                )

                row = {
                    "trial": trial_counter,
                    "bio3d_mode_number": int(bio3d_mode_number),
                    "mode_index_zero_based": int(mode_index),
                    "amplitude_nm": float(amplitude),
                    "padding_nm": float(padding),

                    "score": score_info["score"],
                    "length_rmse_nm": score_info["length_rmse_nm"],
                    "volume_rel_error": score_info["volume_rel_error"],
                    "undersize_penalty": score_info["undersize_penalty"],
                    "max_shortfall_nm": score_info["max_shortfall_nm"],
                    "total_shortfall_nm": score_info["total_shortfall_nm"],
                    "is_undersized": score_info["is_undersized"],

                    "x_shortfall_nm": score_info["x_shortfall_nm"],
                    "y_shortfall_nm": score_info["y_shortfall_nm"],
                    "z_shortfall_nm": score_info["z_shortfall_nm"],
                    "x_error_nm": score_info["x_error_nm"],
                    "y_error_nm": score_info["y_error_nm"],
                    "z_error_nm": score_info["z_error_nm"],

                    "predicted_volume_nm3": score_info["predicted_volume_nm3"],
                    "target_volume_nm3": score_info["target_volume_nm3"],

                    "pred_Lx_nm": float(predicted_lengths[0]),
                    "pred_Ly_nm": float(predicted_lengths[1]),
                    "pred_Lz_nm": float(predicted_lengths[2]),
                    "target_Lx_nm": float(target_box_lengths[0]),
                    "target_Ly_nm": float(target_box_lengths[1]),
                    "target_Lz_nm": float(target_box_lengths[2]),

                    "n_helpers": len(helpers),
                    "mode_start_zero_based": int(mode_index),
                    "n_modes": 1,
                    "first_bio3d_mode": int(bio3d_mode_number),
                    "last_bio3d_mode": int(bio3d_mode_number),

                    "box_lengths_nm": predicted_lengths.tolist(),
                    "box_extents_nm": fit["box_info"]["extents"].tolist(),
                    "box_axes_rows": fit["axes"].tolist(),
                    "box_vectors_nm": fit["box_vectors"].tolist(),
                    "common_align_atoms": int(fit["common_align_atoms"]),
                }

                rows.append(row)

                trial_record = {
                    "row": row,
                    "helper_metadata": helper_metadata,
                }

                trial_records.append(trial_record)

                if best_global is None or row["score"] < best_global["row"]["score"]:
                    best_global = {
                        "row": row,
                        "fit": fit,
                        "helpers": helpers,
                        "helper_metadata": helper_metadata,
                    }

                if args.save_all_trials:
                    amp_label = safe_float_label(amplitude)
                    pad_label = safe_float_label(padding)

                    trial_struct_dir = (
                        structures_dir
                        / "all_trials"
                        / f"mode{bio3d_mode_number:03d}_amp{amp_label}nm_pad{pad_label}nm"
                    )
                    trial_raw_dir = (
                        raw_data_dir
                        / "all_trials"
                        / f"mode{bio3d_mode_number:03d}_amp{amp_label}nm_pad{pad_label}nm"
                    )

                    trial_struct_dir.mkdir(parents=True, exist_ok=True)
                    trial_raw_dir.mkdir(parents=True, exist_ok=True)

                    fit["target_boxed"].save_gro(
                        (trial_struct_dir / "target_common_box.gro").as_posix()
                    )
                    fit["target_boxed"].save_pdb(
                        (trial_struct_dir / "target_common_box.pdb").as_posix()
                    )

                    with open(trial_raw_dir / "trial_summary.json", "w") as f:
                        json.dump(trial_record, f, indent=2)

        print("")

    # -------------------------------------------------------------------------
    # 7. Sort all one-mode trials and select the best x/y/z independently
    # -------------------------------------------------------------------------

    rows_sorted = sorted(rows, key=lambda x: x["score"])

    for rank, row in enumerate(rows_sorted, start=1):
        row["rank"] = rank

    write_sweep_results_csv(
        rows=rows_sorted,
        path=raw_data_dir / "sweep_results.csv",
    )

    with open(raw_data_dir / "sweep_results.json", "w") as f:
        json.dump(rows_sorted, f, indent=2)

    if best_global is None:
        raise RuntimeError("No successful trial was completed.")

    per_axis_best = select_best_per_axis(
        rows=rows,
        target_box_lengths=target_box_lengths,
        undersize_tolerance=args.undersize_tolerance,
    )

    final_lengths = np.array(per_axis_best["final_lengths_nm"], dtype=float)
    final_box = transform_target_to_final_box(
        target=target,
        reference=reference,
        common_align_selection=args.common_align_selection,
        target_fit_selection=args.target_fit_selection,
        center_selection=args.center_selection,
        axes=fixed_axes,
        lengths=final_lengths,
    )

    final_box["target_boxed"].save_gro(
        (structures_dir / "best_per_axis_target_common_box.gro").as_posix()
    )
    final_box["target_boxed"].save_pdb(
        (structures_dir / "best_per_axis_target_common_box.pdb").as_posix()
    )

    np.save(raw_data_dir / "best_per_axis_box_lengths_nm.npy", final_lengths)
    np.save(raw_data_dir / "best_per_axis_box_vectors_nm.npy", final_box["box_vectors"])

    with open(raw_data_dir / "best_per_axis_summary.json", "w") as f:
        json.dump(
            {
                "best_per_axis": per_axis_best,
                "target": str(args.target),
                "reference": str(args.reference),
                "original_gro": str(args.original_gro),
                "nma_dir": str(nma_dir),
                "raw_modes_all": str(raw_modes_all_path),
                "helper_reference": "build_protein_heavy_views(pdb)[0]",
                "common_align_selection": args.common_align_selection,
                "target_fit_selection": args.target_fit_selection,
                "helper_fit_selection": args.helper_fit_selection,
                "center_selection": args.center_selection,
                "target_box_lengths_nm": target_box_lengths.tolist(),
                "target_box_volume_nm3": target_box_volume,
                "fixed_axes_rows": fixed_axes.tolist(),
                "note": (
                    "Per-axis version: each selected normal mode is tested separately "
                    "at each amplitude. The final Lx, Ly, and Lz are selected "
                    "independently from the candidates that are not shorter than "
                    "the original GRO box side. All trials use one fixed PCA box frame."
                ),
            },
            f,
            indent=2,
        )

    # Also save the best single one-mode global candidate for comparison.
    best_global_row = best_global["row"]
    best_global_fit = best_global["fit"]

    best_global_fit["target_boxed"].save_gro(
        (structures_dir / "best_global_one_mode_target_common_box.gro").as_posix()
    )
    best_global_fit["target_boxed"].save_pdb(
        (structures_dir / "best_global_one_mode_target_common_box.pdb").as_posix()
    )

    with open(raw_data_dir / "best_global_one_mode_summary.json", "w") as f:
        json.dump(
            {
                "best_global_one_mode": best_global_row,
                "helper_metadata": best_global["helper_metadata"],
                "note": (
                    "This is only the best single one-mode/amplitude candidate by the "
                    "global score. The recommended result is best_per_axis_summary.json."
                ),
            },
            f,
            indent=2,
        )

    if args.save_best_helpers:
        # Save helpers for the globally best one-mode trial.
        save_helpers(
            helpers=best_global["helpers"],
            metadata=best_global["helper_metadata"],
            outdir=structures_dir / "best_global_one_mode_helpers",
        )

        # Save helpers for each axis-selected mode/amplitude.
        for axis_name, selected_key in [
            ("x", "selected_x"),
            ("y", "selected_y"),
            ("z", "selected_z"),
        ]:
            selected_row = per_axis_best[selected_key]
            selected_helpers, selected_metadata = make_nm_helpers(
                helper_reference=helper_reference,
                nma_modes=nma_modes,
                mode_start=int(selected_row["mode_index_zero_based"]),
                n_modes=1,
                amplitude=float(selected_row["amplitude_nm"]),
            )
            save_helpers(
                helpers=selected_helpers,
                metadata=selected_metadata,
                outdir=structures_dir / "best_per_axis_helpers" / axis_name,
            )

    # -------------------------------------------------------------------------
    # 8. Optional overlay plots
    # -------------------------------------------------------------------------

    if args.plot_overlay:
        overlay_dir = plots_dir / "best_per_axis_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # Plot final target only in the final assembled per-axis box.
        # The selected x/y/z helpers may come from different modes/amplitudes,
        # so the safest visual output here is the final target in the final box.
        plot_overlay_and_box_3d(
            trajs_box_frame=[final_box["target_boxed"]],
            fit_indices_per_traj=[final_box["target_fit_indices"]],
            lengths=final_lengths,
            out_png=overlay_dir / "final_target_box_3d.png",
            max_points_per_structure=args.plot_max_points,
        )

        for plane in ("xy", "xz", "yz"):
            plot_overlay_projection(
                trajs_box_frame=[final_box["target_boxed"]],
                fit_indices_per_traj=[final_box["target_fit_indices"]],
                lengths=final_lengths,
                out_png=overlay_dir / f"final_target_box_{plane}.png",
                plane=plane,
                max_points_per_structure=args.plot_max_points,
            )

    # -------------------------------------------------------------------------
    # 9. Final report
    # -------------------------------------------------------------------------

    print("====================================")
    print("Done")
    print("====================================")
    print("Recommended result: best per-axis box")
    print("")
    print("Selected x-axis candidate:")
    print(
        f"  mode {per_axis_best['selected_x']['bio3d_mode_number']} | "
        f"amp {per_axis_best['selected_x']['amplitude_nm']} nm | "
        f"Lx {per_axis_best['selected_x']['pred_Lx_nm']:.6f} nm | "
        f"x_error {per_axis_best['selected_x']['x_error_nm']:.6f} nm"
    )
    print("Selected y-axis candidate:")
    print(
        f"  mode {per_axis_best['selected_y']['bio3d_mode_number']} | "
        f"amp {per_axis_best['selected_y']['amplitude_nm']} nm | "
        f"Ly {per_axis_best['selected_y']['pred_Ly_nm']:.6f} nm | "
        f"y_error {per_axis_best['selected_y']['y_error_nm']:.6f} nm"
    )
    print("Selected z-axis candidate:")
    print(
        f"  mode {per_axis_best['selected_z']['bio3d_mode_number']} | "
        f"amp {per_axis_best['selected_z']['amplitude_nm']} nm | "
        f"Lz {per_axis_best['selected_z']['pred_Lz_nm']:.6f} nm | "
        f"z_error {per_axis_best['selected_z']['z_error_nm']:.6f} nm"
    )
    print("")
    print(
        "Final per-axis lengths nm:  "
        f"{final_lengths[0]:.6f} "
        f"{final_lengths[1]:.6f} "
        f"{final_lengths[2]:.6f}"
    )
    print(
        "Target GRO lengths nm:      "
        f"{target_box_lengths[0]:.6f} "
        f"{target_box_lengths[1]:.6f} "
        f"{target_box_lengths[2]:.6f}"
    )
    print(
        "Final minus target nm:      "
        f"{final_lengths[0] - target_box_lengths[0]:.6f} "
        f"{final_lengths[1] - target_box_lengths[1]:.6f} "
        f"{final_lengths[2] - target_box_lengths[2]:.6f}"
    )
    print("")
    print("Best global one-mode candidate, for comparison:")
    print(f"  mode:                 {best_global_row['bio3d_mode_number']}")
    print(f"  amplitude nm:         {best_global_row['amplitude_nm']}")
    print(f"  padding nm:           {best_global_row['padding_nm']}")
    print(f"  score:                {best_global_row['score']:.6f}")
    print(f"  max shortfall nm:     {best_global_row['max_shortfall_nm']:.6f}")
    print("")
    print(f"Wrote structures to:    {structures_dir}")
    print(f"Wrote raw data to:      {raw_data_dir}")
    print(f"Wrote plots to:         {plots_dir}")


if __name__ == "__main__":
    main()