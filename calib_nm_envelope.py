#!/usr/bin/env python3
# calib_nm_envelope.py
"""
Step 2 of the box calibration: how does the NM-helper box grow with amplitude?

The reference structure (from calib_required_box.py) is displaced by +/- a
along each of the first K internal normal modes (Bio3D modes 7..6+K). The
box extent along x, y, z of the union of all these structures is recorded
for a grid of amplitudes a. This is the same "target + helpers" envelope that
nm_box_size_predictor_per_axis.py builds, but in the MD box frame and without
looking at the MD box.

Amplitude definition
--------------------
a is the RMS per-atom displacement in nm, so it means the same thing for a
50-residue and a 500-residue protein.
  --scaling uniform : every mode gets a
  --scaling thermal : mode m gets a * sqrt(lambda_7 / lambda_m)
                      (equipartition; softer modes move further)
Both are computed by default so the fit step can compare them.

Inputs
  results/<pdb>/calib/reference_xyz.npy       (from calib_required_box.py)
  results/<pdb>/calib/nma_reference_xyz.npy
  <nma-dir>/raw_modes_all.npy                 (3N, n_modes), Bio3D order
  <nma-dir>/eigenvalues_all.npy

Outputs (results/<pdb>/calib/)
  envelope.npz       grid, and extents[scaling][K] with shape (n_grid, 3)
  nma_features.json  eigenvalue-based flexibility descriptors

Examples
  python calib_nm_envelope.py --pdb 1abc --nma-dir results/1abc/nma
  python calib_nm_envelope.py --pdb 1abc --run-nma          # compute modes first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calib_geometry import kabsch_rotations, rotate_modes, envelope_extents

FIRST_INTERNAL = 6          # zero-based column of Bio3D mode 7


def run_nma(pdb: str, nma_dir: Path, nma_xyz: np.ndarray, n_keep: int) -> None:
    """Compute Bio3D all-atom NMA on the (whole) NMA reference structure."""
    import mdtraj as md
    from traj_utils import build_protein_heavy_views
    from nma_bio3d_structure_modes import run_aanma_r_from_traj

    ref_traj, top_prot, *_ = build_protein_heavy_views(pdb)
    tr = md.Trajectory(nma_xyz[None].astype(np.float32), top_prot)
    run_aanma_r_from_traj(tr, n_modes_keep=n_keep, save_raw_modes_dir=nma_dir)


def nma_features(modes_sel: np.ndarray, eig_sel: np.ndarray, n_atoms: int) -> dict:
    """
    Flexibility descriptors from the selected internal modes.

    nma_msf_K       : sum_m 1/lambda_m / N  (thermal mean-square fluctuation
                      per atom up to kT, in Bio3D units)
    inv_lambda7     : 1/lambda of the softest internal mode
    mode7_localisation : max per-atom displacement / RMS displacement of
                      mode 7 (large = a tail or loop dominates the motion)
    """
    feats = {}
    for k in (5, 10, 20):
        if len(eig_sel) >= k:
            feats[f"nma_msf_{k}"] = float(np.sum(1.0 / eig_sel[:k]) / n_atoms)
    feats["inv_lambda7"] = float(1.0 / eig_sel[0])
    v = modes_sel[:, 0].reshape(n_atoms, 3)
    d = np.linalg.norm(v, axis=1)
    feats["mode7_localisation"] = float(d.max() / np.sqrt(np.mean(d ** 2)))
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--outdir", default="results", type=Path)
    ap.add_argument("--nma-dir", type=Path, default=None,
                    help="Dir with raw_modes_all.npy + eigenvalues_all.npy. "
                         "Default: <outdir>/<pdb>/calib/nma")
    ap.add_argument("--run-nma", action="store_true",
                    help="Run Bio3D NMA first (needs R + bio3d + rpy2).")
    ap.add_argument("--n-modes", type=int, nargs="+", default=[5, 10, 20],
                    help="Numbers of internal modes K to build envelopes for.")
    ap.add_argument("--amp-max", type=float, default=2.0,
                    help="Largest RMS amplitude in the grid (nm).")
    ap.add_argument("--amp-step", type=float, default=0.02)
    args = ap.parse_args()

    pdb = args.pdb.strip().lower()
    calib = args.outdir / pdb / "calib"
    nma_dir = args.nma_dir or (calib / "nma")

    ref = np.load(calib / "reference_xyz.npy")
    nma_xyz = np.load(calib / "nma_reference_xyz.npy")
    n = ref.shape[0]
    kmax = max(args.n_modes)

    if args.run_nma:
        run_nma(pdb, nma_dir, nma_xyz, n_keep=kmax)

    modes_path = nma_dir / "raw_modes_all.npy"
    eig_path = nma_dir / "eigenvalues_all.npy"
    if not modes_path.exists():
        raise FileNotFoundError(f"{modes_path} missing. Use --run-nma or --nma-dir.")

    # mmap: raw_modes_all can be (3N x 3N) and several GB
    modes_all = np.load(modes_path, mmap_mode="r")
    if modes_all.shape[0] != 3 * n:
        raise ValueError(f"raw_modes_all has {modes_all.shape[0]} rows; reference has {n} atoms "
                         f"({3 * n} DOF). Different atom set.")
    if modes_all.shape[1] < FIRST_INTERNAL + kmax:
        raise ValueError(f"raw_modes_all has only {modes_all.shape[1]} modes, need {FIRST_INTERNAL + kmax}")

    modes = np.array(modes_all[:, FIRST_INTERNAL:FIRST_INTERNAL + kmax], dtype=np.float64)
    eig = np.load(eig_path)[FIRST_INTERNAL:FIRST_INTERNAL + kmax].astype(float)

    # Rotate modes from the NMA structure's frame into the reference frame.
    R, _, _ = kabsch_rotations(nma_xyz[None], ref)
    modes = rotate_modes(modes, R[0])

    grid = np.round(np.arange(0.0, args.amp_max + 1e-9, args.amp_step), 6)
    save = {"grid": grid, "n_modes": np.array(args.n_modes)}
    for scaling in ("uniform", "thermal"):
        for k in args.n_modes:
            env = envelope_extents(ref, modes[:, :k], grid, eig[:k], scaling)
            save[f"{scaling}_K{k}"] = env
            print(f"[{scaling:7s} K={k:2d}] extent at a=0: {np.round(env[0], 2)}  "
                  f"a={grid[len(grid) // 4]:.2f}: {np.round(env[len(grid) // 4], 2)}")

    np.savez_compressed(calib / "envelope.npz", **save)

    feats = nma_features(modes, eig, n)
    feats["pdb"] = pdb
    (calib / "nma_features.json").write_text(json.dumps(feats, indent=2))
    print(f"written {calib / 'envelope.npz'} and nma_features.json")


if __name__ == "__main__":
    main()
