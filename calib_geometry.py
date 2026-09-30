# calib_geometry.py
"""
Pure-NumPy geometry used by the box-calibration scripts.

Kept free of MDTraj / R so it can be unit-tested anywhere
(see test_calib.py). All coordinates are in nm.

Conventions
-----------
xyz      : (n_frames, n_atoms, 3) or (n_atoms, 3)
box      : (n_frames, 3, 3) box vectors as rows (MDTraj unitcell_vectors)
extents  : (n_frames, 3) max-min of coordinates along x, y, z
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Periodic unwrapping
# ---------------------------------------------------------------------------

def _min_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Minimum-image correction of difference vectors d (F, M, 3) with box
    vectors box (F, 3, 3). Exact for orthorhombic boxes; for triclinic boxes
    it is exact for vectors much shorter than the box, which is all we need
    for neighbouring atoms along a chain.
    """
    inv = np.linalg.inv(box)                     # (F, 3, 3)
    frac = np.einsum("fmi,fij->fmj", d, inv)     # fractional coords
    shift = np.round(frac)
    return d - np.einsum("fmi,fij->fmj", shift, box)


def make_whole(
    xyz: np.ndarray,
    box: np.ndarray,
    chain_starts: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    Undo periodic wrapping for a protein (possibly several chains).

    Step 1 (within the protein): walk the atoms in topology order and apply
    the minimum-image convention to each consecutive difference vector.
    Consecutive heavy atoms in a protein are < 0.5 nm apart, far below half
    a box length, so this reconstructs each chain exactly.

    Step 2 (between chains): each chain after the first is shifted by the
    lattice vector (among the 27 neighbouring images, including no shift)
    that makes the bounding box of the chains placed so far smallest, i.e.
    the most compact assembly. This fixes chains whose junction atoms happen
    to be more than half a box apart. (A centroid-distance rule fails for
    elongated complexes whose chain centroids are > L/2 apart.)

    Returns float64 coordinates, same shape as xyz.
    """
    x = np.asarray(xyz, dtype=np.float64)
    single = x.ndim == 2
    if single:
        x = x[None]
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 2:
        box = box[None]
    if box.shape[0] == 1 and x.shape[0] > 1:
        box = np.repeat(box, x.shape[0], axis=0)

    d = np.diff(x, axis=1)
    d = _min_image(d, box)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    out[:, 1:] = x[:, :1] + np.cumsum(d, axis=1)

    if chain_starts is not None and len(chain_starts) > 1:
        starts = sorted(chain_starts) + [x.shape[1]]
        lattice = np.array(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=np.float64,
        )                                                     # (27, 3)
        shifts = np.einsum("si,fij->fsj", lattice, box)       # (F, 27, 3)
        pmin = out[:, starts[0]:starts[1]].min(axis=1)
        pmax = out[:, starts[0]:starts[1]].max(axis=1)
        for c in range(1, len(starts) - 1):
            sl = slice(starts[c], starts[c + 1])
            cmin = out[:, sl].min(axis=1)
            cmax = out[:, sl].max(axis=1)
            hi = np.maximum(pmax[:, None, :], cmax[:, None, :] + shifts)
            lo = np.minimum(pmin[:, None, :], cmin[:, None, :] + shifts)
            size = (hi - lo).sum(axis=2)                      # (F, 27)
            best = np.argmin(size, axis=1)
            # Chains in contact overlap in projection on every axis. If the
            # chain walk of step 1 already gives that, trust it (it is exact
            # whenever the junction atoms are < L/2 apart).
            overlap = np.all((cmin <= pmax + 0.5) & (cmax >= pmin - 0.5), axis=1)
            best[overlap] = 13                                # index of (0,0,0)
            s = shifts[np.arange(len(best)), best]            # (F, 3)
            out[:, sl] += s[:, None, :]
            pmin = np.minimum(pmin, cmin + s)
            pmax = np.maximum(pmax, cmax + s)

    return out[0] if single else out


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def kabsch_rotations(
    mobile: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal rotations superposing mobile frames onto ref.

    mobile : (F, M, 3), ref : (M, 3)
    Returns R (F, 3, 3), mobile centroids (F, 3), ref centroid (3,)
    such that aligned = (x - c_mob) @ R + c_ref.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    c_mob = mobile.mean(axis=1)
    c_ref = ref.mean(axis=0)
    P = mobile - c_mob[:, None, :]
    Q = ref - c_ref
    H = np.einsum("fmi,mj->fij", P, Q)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", U, Vt)))
    D = np.zeros_like(H)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("fij,fjk,fkl->fil", U, D, Vt)
    return R, c_mob, c_ref


def superpose(
    xyz: np.ndarray,
    ref: np.ndarray,
    align_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Superpose every frame of xyz (F, N, 3) onto ref (N, 3) using align_idx."""
    x = np.asarray(xyz, dtype=np.float64)
    single = x.ndim == 2
    if single:
        x = x[None]
    idx = np.arange(x.shape[1]) if align_idx is None else np.asarray(align_idx)
    R, c_mob, c_ref = kabsch_rotations(x[:, idx], np.asarray(ref)[idx])
    out = np.einsum("fni,fij->fnj", x - c_mob[:, None, :], R) + c_ref
    return out[0] if single else out


def rotate_modes(modes_3n: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate mode vectors (3N, K) by a single rotation R (3, 3) using the same
    row-vector convention as superpose(): v_new = v @ R for every atom.
    """
    m = np.asarray(modes_3n, dtype=np.float64)
    n3, k = m.shape
    v = m.T.reshape(k, n3 // 3, 3)
    v = v @ R
    return v.reshape(k, n3).T


# ---------------------------------------------------------------------------
# Shape descriptors
# ---------------------------------------------------------------------------

def extents(xyz: np.ndarray) -> np.ndarray:
    """max - min along x, y, z. (F, N, 3) -> (F, 3); (N, 3) -> (3,)"""
    x = np.asarray(xyz)
    return x.max(axis=-2) - x.min(axis=-2)


def max_diameter(xyz: np.ndarray, max_points: int = 4000, seed: int = 0) -> float:
    """
    Largest atom-atom distance of one structure (N, 3). The diameter is
    always realised by two points on the convex hull, so hull vertices are
    used when scipy is available; otherwise a random subsample is taken.
    """
    x = np.asarray(xyz, dtype=np.float64)
    try:
        from scipy.spatial import ConvexHull
        x = x[ConvexHull(x).vertices]
    except Exception:
        if x.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            x = x[rng.choice(x.shape[0], max_points, replace=False)]
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


def gyration_features(xyz: np.ndarray, masses: np.ndarray | None = None) -> dict:
    """
    Radius of gyration and shape from the gyration tensor of one structure.

    asphericity  = l1 - (l2 + l3)/2 normalised by Rg^2 (0 = sphere)
    axis ratios  = sqrt(l1/l3), sqrt(l2/l3)   (l1 >= l2 >= l3)
    """
    x = np.asarray(xyz, dtype=np.float64)
    w = np.ones(len(x)) if masses is None else np.asarray(masses, dtype=float)
    w = w / w.sum()
    c = (x * w[:, None]).sum(0)
    X = x - c
    S = (X * w[:, None]).T @ X
    lam = np.sort(np.linalg.eigvalsh(S))[::-1]
    rg2 = lam.sum()
    return {
        "rg_nm": float(np.sqrt(rg2)),
        "asphericity": float((lam[0] - 0.5 * (lam[1] + lam[2])) / rg2),
        "axis_ratio_1_3": float(np.sqrt(lam[0] / lam[2])),
        "axis_ratio_2_3": float(np.sqrt(lam[1] / lam[2])),
    }


def terminal_coil_lengths(ss_per_chain: list[str], coil_chars: str = "C ") -> dict:
    """
    ss_per_chain: one simplified-DSSP string per chain ('H','E','C').
    Returns total and maximum number of consecutive coil residues at the chain
    termini (N- and C-terminal tails), plus overall coil fraction.
    """
    tails = []
    n_coil = 0
    n_tot = 0
    for s in ss_per_chain:
        n_tot += len(s)
        n_coil += sum(ch in coil_chars for ch in s)
        lead = len(s) - len(s.lstrip(coil_chars))
        trail = len(s) - len(s.rstrip(coil_chars))
        if lead == len(s):          # all coil
            trail = 0
        tails += [lead, trail]
    return {
        "terminal_coil_total": int(sum(tails)),
        "terminal_coil_max": int(max(tails) if tails else 0),
        "coil_fraction": float(n_coil / n_tot) if n_tot else float("nan"),
    }


# ---------------------------------------------------------------------------
# NM envelope
# ---------------------------------------------------------------------------

def mode_amplitudes(
    rms_amplitude: float,
    eigvals: np.ndarray | None,
    scaling: str = "thermal",
) -> np.ndarray:
    """
    Per-mode amplitude, expressed as RMS per-atom displacement (nm).

    uniform : every mode gets rms_amplitude
    thermal : mode m gets rms_amplitude * sqrt(lambda_first / lambda_m),
              i.e. equipartition: softer modes move further.
    """
    if scaling == "uniform" or eigvals is None:
        k = 1 if eigvals is None else len(eigvals)
        return np.full(k, float(rms_amplitude))
    ev = np.asarray(eigvals, dtype=float)
    if np.any(ev <= 0):
        raise ValueError("thermal scaling needs positive eigenvalues (drop trivial modes)")
    return float(rms_amplitude) * np.sqrt(ev[0] / ev)


def envelope_extents(
    ref_xyz: np.ndarray,
    modes_3n: np.ndarray,
    rms_amplitudes: np.ndarray,
    eigvals: np.ndarray | None = None,
    scaling: str = "thermal",
) -> np.ndarray:
    """
    Box extent along x, y, z of the union of the reference structure and the
    +/- displaced structures along each mode, for a grid of amplitudes.

    ref_xyz        : (N, 3)
    modes_3n       : (3N, K) mode vectors (any norm; normalised here)
    rms_amplitudes : (A,) grid of amplitudes (RMS per-atom displacement, nm)

    Returns (A, 3).
    """
    ref = np.asarray(ref_xyz, dtype=np.float64)
    n = ref.shape[0]
    M = np.asarray(modes_3n, dtype=np.float64)
    if M.shape[0] != 3 * n:
        raise ValueError(f"modes have {M.shape[0]} rows, expected {3 * n}")
    M = M / np.linalg.norm(M, axis=0, keepdims=True)
    # unit 3N vector -> RMS per-atom displacement 1/sqrt(N); rescale so that
    # amplitude a gives RMS displacement a.
    U = (M * np.sqrt(n)).T.reshape(-1, n, 3)            # (K, N, 3)

    ref_min = ref.min(0)
    ref_max = ref.max(0)
    out = np.empty((len(rms_amplitudes), 3))
    for i, a in enumerate(rms_amplitudes):
        amps = mode_amplitudes(a, eigvals, scaling)
        if len(amps) == 1 and U.shape[0] > 1:
            amps = np.full(U.shape[0], amps[0])
        disp = U * amps[:, None, None]                   # (K, N, 3)
        plus = ref[None] + disp
        minus = ref[None] - disp
        hi = np.maximum(plus.max(axis=(0, 1)), minus.max(axis=(0, 1)))
        lo = np.minimum(plus.min(axis=(0, 1)), minus.min(axis=(0, 1)))
        out[i] = np.maximum(hi, ref_max) - np.minimum(lo, ref_min)
    return out


def solve_amplitude(
    grid: np.ndarray,
    env: np.ndarray,
    required: float,
) -> float:
    """
    Smallest amplitude a with env(a) >= required, by linear interpolation on
    the grid and linear extrapolation beyond its end. env must be
    non-decreasing (it is, for a union envelope). Returns 0 if the starting
    structure already covers the requirement and inf if env is flat.
    """
    g = np.asarray(grid, dtype=float)
    e = np.maximum.accumulate(np.asarray(env, dtype=float))
    if required <= e[0]:
        return 0.0
    hit = np.nonzero(e >= required)[0]
    if hit.size:
        j = hit[0]
        g0, g1, e0, e1 = g[j - 1], g[j], e[j - 1], e[j]
        return float(g0 + (required - e0) * (g1 - g0) / (e1 - e0))
    # extrapolate from the last two distinct points
    slope = (e[-1] - e[-2]) / (g[-1] - g[-2])
    if slope <= 0:
        return float("inf")
    return float(g[-1] + (required - e[-1]) / slope)


def envelope_at(grid: np.ndarray, env: np.ndarray, a: float) -> float:
    """env evaluated at amplitude a (linear interpolation / extrapolation)."""
    g = np.asarray(grid, dtype=float)
    e = np.maximum.accumulate(np.asarray(env, dtype=float))
    if a <= g[-1]:
        return float(np.interp(a, g, e))
    slope = (e[-1] - e[-2]) / (g[-1] - g[-2])
    return float(e[-1] + slope * (a - g[-1]))
