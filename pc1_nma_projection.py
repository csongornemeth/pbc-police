# pc1_nma_projection.py

import numpy as np
import mdtraj as md
from pathlib import Path


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a 1D vector to unit norm.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    norm = np.linalg.norm(v)

    if norm < eps:
        raise ValueError("Cannot normalize vector with near-zero norm.")

    return v / norm


def normalize_mode_columns(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize columns of a mode matrix.

    Parameters
    ----------
    V : np.ndarray
        Shape (3N, n_modes)

    Returns
    -------
    V_norm : np.ndarray
        Shape (3N, n_modes), each column unit-normalized.
    """
    V = np.asarray(V, dtype=float)

    if V.ndim != 2:
        raise ValueError(f"Expected V to be 2D, got shape {V.shape}")

    norms = np.linalg.norm(V, axis=0)

    if np.any(norms < eps):
        bad = np.where(norms < eps)[0]
        raise ValueError(f"Cannot normalize near-zero mode columns: {bad}")

    return V / norms[None, :]


def project_pc_onto_nma_modes(
    pc1_vector: np.ndarray,
    nma_modes: np.ndarray,
    mode_start: int = 6,
    n_modes: int = 5,
):
    """
    Project PC1 onto selected NMA modes.

    Equivalent R logic
    ------------------
    pc <- pca$au[, 1]
    pc <- pc / sqrt(sum(pc^2))

    V <- nma$modes[, 7:11]
    V <- apply(V, 2, function(v) v / sqrt(sum(v^2)))

    coef <- as.numeric(crossprod(V, pc))
    captured <- sum(coef^2)

    dr_dir <- as.numeric(V %*% coef)
    dr_dir <- dr_dir / sqrt(sum(dr_dir^2))

    Parameters
    ----------
    pc1_vector : np.ndarray
        PC1 eigenvector, shape (3N,) or compatible.

    nma_modes : np.ndarray
        Raw NMA mode matrix, shape (3N, n_total_modes).
        This should be the unweighted Bio3D matrix equivalent to nma$modes.

    mode_start : int
        Zero-based start index.
        For Bio3D mode 7, use mode_start=6.

    n_modes : int
        Number of modes to use.
        For Bio3D modes 7:11, use n_modes=5.

    Returns
    -------
    result : dict
        Contains selected normalized modes, coefficients, captured fraction,
        and the normalized reconstructed direction.
    """

    pc = normalize_vector(pc1_vector)

    nma_modes = np.asarray(nma_modes, dtype=float)

    if nma_modes.ndim != 2:
        raise ValueError(
            f"Expected nma_modes shape (3N, n_modes), got {nma_modes.shape}"
        )

    if nma_modes.shape[0] != pc.shape[0]:
        raise ValueError(
            "PC1 and NMA mode matrix have incompatible dimensions:\n"
            f"  pc1_vector length: {pc.shape[0]}\n"
            f"  nma_modes rows:     {nma_modes.shape[0]}"
        )

    mode_stop = mode_start + n_modes

    if mode_stop > nma_modes.shape[1]:
        raise ValueError(
            f"Requested modes {mode_start}:{mode_stop}, "
            f"but NMA matrix only has {nma_modes.shape[1]} columns."
        )

    # Equivalent to nma$modes[, 7:11] in R if mode_start=6
    V = nma_modes[:, mode_start:mode_stop]

    # Normalize each NMA mode column
    V = normalize_mode_columns(V)

    # Equivalent to crossprod(V, pc), shape (n_modes,)
    coef = V.T @ pc

    # Since pc and each mode are unit vectors, this is the fraction of PC1
    # captured by the subspace spanned by the selected NMA modes.
    captured = float(np.sum(coef ** 2))

    # Combined NMA direction that best represents PC1 in this selected subspace
    dr_dir = V @ coef
    dr_dir = normalize_vector(dr_dir)

    return {
        "pc1_normalized": pc,
        "V_normalized": V,
        "coef": coef,
        "captured": captured,
        "dr_dir": dr_dir,
        "mode_indices_zero_based": np.arange(mode_start, mode_stop),
        "bio3d_mode_numbers": np.arange(mode_start, mode_stop) + 1,
    }


def make_pc1_nma_endpoint_coordinates(
    xyz0: np.ndarray,
    dr_dir: np.ndarray,
    pc1_scores: np.ndarray,
):
    """
    Build plus/minus endpoint structures along the NMA-reconstructed PC1 direction.

    Equivalent R logic
    ------------------
    A_pc <- (max(pca$z[,1]) - min(pca$z[,1])) / 2
    dr_scaled <- A_pc * dr_dir

    xyz_plus  <- xyz0 + matrix(dr_scaled, ncol = 3, byrow = TRUE)
    xyz_minus <- xyz0 - matrix(dr_scaled, ncol = 3, byrow = TRUE)

    Parameters
    ----------
    xyz0 : np.ndarray
        Reference coordinates, shape (N, 3).

    dr_dir : np.ndarray
        Normalized displacement direction, shape (3N,).

    pc1_scores : np.ndarray
        PCA projections/scores along PC1, equivalent to pca$z[, 1].
        Units must match xyz0.

    Returns
    -------
    xyz_minus, xyz_plus, A_pc, dr_scaled_atoms
    """

    xyz0 = np.asarray(xyz0, dtype=float)

    if xyz0.ndim != 2 or xyz0.shape[1] != 3:
        raise ValueError(f"Expected xyz0 shape (N, 3), got {xyz0.shape}")

    n_atoms = xyz0.shape[0]

    dr_dir = np.asarray(dr_dir, dtype=float).reshape(-1)

    if dr_dir.shape[0] != n_atoms * 3:
        raise ValueError(
            "dr_dir length does not match xyz0:\n"
            f"  dr_dir length: {dr_dir.shape[0]}\n"
            f"  expected:      {n_atoms * 3}"
        )

    pc1_scores = np.asarray(pc1_scores, dtype=float).reshape(-1)

    A_pc = float((np.max(pc1_scores) - np.min(pc1_scores)) / 2.0)

    dr_scaled = A_pc * dr_dir
    dr_scaled_atoms = dr_scaled.reshape(n_atoms, 3)

    xyz_plus = xyz0 + dr_scaled_atoms
    xyz_minus = xyz0 - dr_scaled_atoms

    return xyz_minus, xyz_plus, A_pc, dr_scaled_atoms


def save_endpoint_structures(
    reference_traj: md.Trajectory,
    xyz_minus: np.ndarray,
    xyz_plus: np.ndarray,
    outdir: str | Path,
    prefix: str = "pc1_nma",
):
    """
    Save endpoint structures using the topology of a reference MDTraj trajectory.

    Important
    ---------
    MDTraj coordinates are in nm.
    Therefore xyz_minus and xyz_plus must also be in nm.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if reference_traj.n_frames < 1:
        raise ValueError("reference_traj must contain at least one frame.")

    xyz_minus = np.asarray(xyz_minus, dtype=float)
    xyz_plus = np.asarray(xyz_plus, dtype=float)

    expected_shape = (reference_traj.n_atoms, 3)

    if xyz_minus.shape != expected_shape:
        raise ValueError(
            f"xyz_minus shape mismatch: got {xyz_minus.shape}, "
            f"expected {expected_shape}"
        )

    if xyz_plus.shape != expected_shape:
        raise ValueError(
            f"xyz_plus shape mismatch: got {xyz_plus.shape}, "
            f"expected {expected_shape}"
        )

    # Make clean one-frame copies with the same topology.
    traj_minus = md.Trajectory(
        xyz=xyz_minus[None, :, :],
        topology=reference_traj.topology,
    )

    traj_plus = md.Trajectory(
        xyz=xyz_plus[None, :, :],
        topology=reference_traj.topology,
    )

    minus_path = outdir / f"{prefix}_minus.pdb"
    plus_path = outdir / f"{prefix}_plus.pdb"

    traj_minus.save(str(minus_path))
    traj_plus.save(str(plus_path))

    print(f"[SAVE] minus endpoint: {minus_path}")
    print(f"[SAVE] plus endpoint:  {plus_path}")

    return minus_path, plus_path