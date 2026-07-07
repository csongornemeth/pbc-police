"""
python bob_the_nm_boxer.py \
  --pdb 7lak \
  --target target_7lak.pdb \
  --reference test.pdb \
  --original-gro prep.gro \
  --run-nma \
  --mode-start 6 \
  --n-modes 10 \
  --skip-existing-nma \
  --amplitudes 0 0.5 1 1.5 2 2.4 2.5 3 3.5 4 4.5 5 5.5\
  --paddings 0.7\
  --target-fit-selection "(protein and not element H) or (resname IBI and not element H)"\
  --plot-overlay\
  --save-all-trials\
  --save-best-helpers
  --volume-weight 1.0
"""
#!/usr/bin/env python3
# nm_box_size_predictor.py

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

def atom_key(atom):
    res = atom.residue
    chain = res.chain

    return (
        chain.index,
        res.resSeq,
        res.name,
        atom.name,
    )


def build_atom_key_map(traj: md.Trajectory, selection: str) -> dict:
    indices = traj.topology.select(selection)
    out = {}

    for idx in indices:
        atom = traj.topology.atom(int(idx))
        key = atom_key(atom)

        if key in out:
            raise ValueError(f"Duplicate atom key found: {key}")

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

    axes = pca_axes(point_cloud)

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
    predicted_lengths,
    target_lengths,
    volume_weight,
    undersize_tolerance=0.0,
    reject_undersized=True,
):
    predicted_lengths = np.asarray(predicted_lengths, dtype=float)
    target_lengths = np.asarray(target_lengths, dtype=float)

    predicted_sorted = np.sort(predicted_lengths)
    target_sorted = np.sort(target_lengths)

    shortfall = np.maximum(target_sorted - predicted_sorted, 0.0)
    max_shortfall = float(np.max(shortfall))

    length_rmse = float(
        np.sqrt(np.mean((predicted_sorted - target_sorted) ** 2))
    )

    predicted_volume = float(np.prod(predicted_lengths))
    target_volume = float(np.prod(target_lengths))

    volume_rel_error = float(
        abs(predicted_volume - target_volume) / target_volume
    )

    is_undersized = bool(np.any(shortfall > undersize_tolerance))

    if reject_undersized and is_undersized:
        score = float("inf")
    else:
        score = float(length_rmse + float(volume_weight) * volume_rel_error)

    return {
        "score": score,
        "length_rmse_nm": length_rmse,
        "volume_rel_error": volume_rel_error,
        "max_shortfall_nm": max_shortfall,
        "is_undersized": is_undersized,
        "predicted_volume_nm3": predicted_volume,
        "target_volume_nm3": target_volume,
        "predicted_lengths_sorted_nm": predicted_sorted.tolist(),
        "target_lengths_sorted_nm": target_sorted.tolist(),
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
        "amplitude_nm",
        "padding_nm",
        "score",
        "length_rmse_nm",
        "volume_rel_error",
        "max_shortfall_nm",
        "is_undersized",
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
    # 5. Sweep amplitudes and paddings
    # -------------------------------------------------------------------------

    rows = []
    trial_records = []
    best = None

    trial_counter = 0

    for amplitude in args.amplitudes:
        print(f"[Amplitude] {amplitude} nm")

        helpers, helper_metadata = make_nm_helpers(
            helper_reference=helper_reference,
            nma_modes=nma_modes,
            mode_start=args.mode_start,
            n_modes=args.n_modes,
            amplitude=amplitude,
        )

        print(f"  generated helpers: {len(helpers)}")

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
            )

            predicted_lengths = fit["lengths"]

            score_info = score_box(
                predicted_lengths=predicted_lengths,
                target_lengths=target_box_lengths,
                volume_weight=args.volume_weight,
            )

            row = {
                "trial": trial_counter,
                "amplitude_nm": float(amplitude),
                "padding_nm": float(padding),
                "score": score_info["score"],
                "length_rmse_nm": score_info["length_rmse_nm"],
                "volume_rel_error": score_info["volume_rel_error"],
                "max_shortfall_nm": score_info["max_shortfall_nm"],
                "is_undersized": score_info["is_undersized"],
                "predicted_volume_nm3": score_info["predicted_volume_nm3"],
                "target_volume_nm3": score_info["target_volume_nm3"],
                "pred_Lx_nm": float(predicted_lengths[0]),
                "pred_Ly_nm": float(predicted_lengths[1]),
                "pred_Lz_nm": float(predicted_lengths[2]),
                "target_Lx_nm": float(target_box_lengths[0]),
                "target_Ly_nm": float(target_box_lengths[1]),
                "target_Lz_nm": float(target_box_lengths[2]),
                "n_helpers": len(helpers),
                "mode_start_zero_based": int(args.mode_start),
                "n_modes": int(args.n_modes),
                "first_bio3d_mode": int(args.mode_start + 1),
                "last_bio3d_mode": int(args.mode_start + args.n_modes),
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

            if best is None or row["score"] < best["row"]["score"]:
                best = {
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
                    / f"amp{amp_label}nm_pad{pad_label}nm"
                )
                trial_raw_dir = (
                    raw_data_dir
                    / "all_trials"
                    / f"amp{amp_label}nm_pad{pad_label}nm"
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
    # 6. Sort results and write sweep data
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

    if best is None:
        raise RuntimeError("No successful trial was completed.")

    best_row = best["row"]
    best_fit = best["fit"]

    # -------------------------------------------------------------------------
    # 7. Save best boxed target and raw data
    # -------------------------------------------------------------------------

    best_fit["target_boxed"].save_gro(
        (structures_dir / "best_target_common_box.gro").as_posix()
    )
    best_fit["target_boxed"].save_pdb(
        (structures_dir / "best_target_common_box.pdb").as_posix()
    )

    np.save(raw_data_dir / "best_box_axes_rows.npy", best_fit["axes"])
    np.save(raw_data_dir / "best_box_lengths_nm.npy", best_fit["lengths"])
    np.save(raw_data_dir / "best_box_vectors_nm.npy", best_fit["box_vectors"])

    best_summary = {
        "best": best_row,
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
        "helper_metadata": best["helper_metadata"],
        "note": (
            "Version A prototype: individual selected NMs, plus/minus helpers, "
            "uniform amplitude for all selected modes. Padding is one-sided, "
            "so final length = fitted extent + 2 * padding."
        ),
    }

    with open(raw_data_dir / "best_box_summary.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    if args.save_best_helpers:
        save_helpers(
            helpers=best["helpers"],
            metadata=best["helper_metadata"],
            outdir=structures_dir / "best_helpers",
        )

    # -------------------------------------------------------------------------
    # 8. Optional overlay plots
    # -------------------------------------------------------------------------

    if args.plot_overlay:
        overlay_dir = plots_dir / "best_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        all_boxed = [best_fit["target_boxed"]] + best_fit["helpers_boxed"]
        fit_indices_per_traj = [
            best_fit["target_fit_indices"]
        ] + best_fit["helper_fit_indices"]

        plot_overlay_and_box_3d(
            trajs_box_frame=all_boxed,
            fit_indices_per_traj=fit_indices_per_traj,
            lengths=best_fit["lengths"],
            out_png=overlay_dir / "overlay_box_3d.png",
            max_points_per_structure=args.plot_max_points,
        )

        for plane in ("xy", "xz", "yz"):
            plot_overlay_projection(
                trajs_box_frame=all_boxed,
                fit_indices_per_traj=fit_indices_per_traj,
                lengths=best_fit["lengths"],
                out_png=overlay_dir / f"overlay_box_{plane}.png",
                plane=plane,
                max_points_per_structure=args.plot_max_points,
            )

    # -------------------------------------------------------------------------
    # 9. Final report
    # -------------------------------------------------------------------------

    print("====================================")
    print("Done")
    print("====================================")
    print(f"Best amplitude nm:     {best_row['amplitude_nm']}")
    print(f"Best padding nm:       {best_row['padding_nm']}")
    print(f"Best score:            {best_row['score']:.6f}")
    print(f"Length RMSE nm:        {best_row['length_rmse_nm']:.6f}")
    print(f"Volume rel error:      {best_row['volume_rel_error']:.6f}")
    print(
        "Predicted lengths nm:  "
        f"{best_row['pred_Lx_nm']:.6f} "
        f"{best_row['pred_Ly_nm']:.6f} "
        f"{best_row['pred_Lz_nm']:.6f}"
    )
    print(
        "Target lengths nm:     "
        f"{best_row['target_Lx_nm']:.6f} "
        f"{best_row['target_Ly_nm']:.6f} "
        f"{best_row['target_Lz_nm']:.6f}"
    )
    print("")
    print(f"Wrote structures to:   {structures_dir}")
    print(f"Wrote raw data to:     {raw_data_dir}")
    print(f"Wrote plots to:        {plots_dir}")


if __name__ == "__main__":
    main()