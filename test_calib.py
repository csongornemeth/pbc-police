#!/usr/bin/env python3
# test_calib.py
"""
Self-tests for the box-calibration code on synthetic data.
Needs only numpy / scipy / scikit-learn / pandas / matplotlib (no MDTraj, R).

  python test_calib.py            # unit tests + synthetic end-to-end fit
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from calib_geometry import (
    make_whole, superpose, kabsch_rotations, rotate_modes, extents,
    envelope_extents, solve_amplitude, envelope_at, gyration_features,
    terminal_coil_lengths, max_diameter,
)

HERE = Path(__file__).resolve().parent


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def synthetic_protein(n_atoms: int, rng, step: float = 0.15) -> np.ndarray:
    """Compact self-avoiding-ish random walk (nm)."""
    x = np.zeros((n_atoms, 3))
    for i in range(1, n_atoms):
        for _ in range(20):
            d = rng.normal(size=3)
            d *= step / np.linalg.norm(d)
            # weak pull to the centroid keeps it globular
            d -= 0.02 * (x[i - 1] - x[:i].mean(0))
            cand = x[i - 1] + d
            if i < 3 or np.min(np.linalg.norm(x[: i - 2] - cand, axis=1)) > 0.1:
                break
        x[i] = cand
    return x


def synthetic_modes(xyz: np.ndarray, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Smooth, orthonormal, rigid-body-free displacement fields + eigenvalues."""
    n = len(xyz)
    c = xyz - xyz.mean(0)
    rigid = [np.tile(np.eye(3)[a], n) for a in range(3)]
    for a in range(3):
        e = np.eye(3)[a]
        rigid.append(np.cross(e, c).reshape(-1))
    fields = []
    for _ in range(k):
        w = rng.normal(size=(8, 3))
        centres = c[rng.choice(n, 8, replace=False)]
        d = np.exp(-((c[:, None, :] - centres[None]) ** 2).sum(-1) / 1.0)   # (n, 8)
        fields.append((d @ w).reshape(-1))
    B = np.array(rigid + fields).T
    Q, _ = np.linalg.qr(B)
    modes = Q[:, 6:6 + k]
    eig = np.sort(rng.uniform(0.5, 5.0, size=k))
    return modes, eig


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_make_whole(rng):
    for tric, reverse_b in ((False, False), (True, False), (False, True), (True, True)):
        # two chains in contact: split one globule; optionally reverse chain
        # B's atom order so its first atom is far from A's last atom
        full = synthetic_protein(700, rng)
        a, b = full[:400], full[400:]
        if reverse_b:
            b = b[::-1]
        xyz = np.concatenate([a, b])
        frames = np.stack([xyz @ random_rotation(rng) + rng.uniform(-3, 3, 3) for _ in range(5)])
        box = np.diag([4.0, 4.5, 5.0])
        if tric:
            box = np.array([[5.0, 0, 0], [0, 5.0, 0], [2.5, 2.5, 3.5355]])   # dodecahedron-like
        boxes = np.repeat(box[None], len(frames), axis=0)
        # wrap into the unit cell
        frac = np.einsum("fni,ij->fnj", frames, np.linalg.inv(box))
        wrapped = np.einsum("fni,ij->fnj", frac - np.floor(frac), box)
        whole = make_whole(wrapped, boxes, chain_starts=[0, 400])
        err = np.abs(extents(whole) - extents(frames)).max()
        # whole structure equals original up to one lattice translation
        shift = whole - frames
        assert np.allclose(shift, shift[:, :1, :], atol=1e-6), "internal geometry changed"
        assert err < 1e-6, err
    print("ok  make_whole (orthorhombic + triclinic, 2 chains)")


def test_superpose(rng):
    x = synthetic_protein(200, rng)
    R = random_rotation(rng)
    y = x @ R + np.array([1.0, -2.0, 0.5])
    back = superpose(y[None], x)[0]
    assert np.abs(back - x).max() < 1e-8
    # rotate_modes must follow the same convention
    modes, _ = synthetic_modes(x, 3, rng)
    Rk, _, _ = kabsch_rotations(x[None], y)
    rm = rotate_modes(modes, Rk[0])
    expected = (modes.T.reshape(3, -1, 3) @ R).reshape(3, -1).T
    assert np.abs(rm - expected).max() < 1e-8
    print("ok  superpose + rotate_modes")


def test_envelope(rng):
    x = synthetic_protein(300, rng)
    modes, eig = synthetic_modes(x, 10, rng)
    grid = np.linspace(0, 1.0, 51)
    for scaling in ("uniform", "thermal"):
        env = envelope_extents(x, modes, grid, eig, scaling)
        assert np.allclose(env[0], extents(x))
        assert np.all(np.diff(env, axis=0) >= -1e-12), "envelope not monotone"
        for a in range(3):
            target = env[0, a] + 0.6 * (env[-1, a] - env[0, a])
            A = solve_amplitude(grid, env[:, a], target)
            assert abs(envelope_at(grid, env[:, a], A) - target) < 1e-9
        # extrapolation beyond the grid
        A = solve_amplitude(grid, env[:, 0], env[-1, 0] + 0.5)
        assert A > grid[-1] and np.isfinite(A)
    assert solve_amplitude(grid, env[:, 0], 0.0) == 0.0
    print("ok  envelope monotone, solve_amplitude inverts envelope_at")


def test_features(rng):
    x = synthetic_protein(300, rng)
    g = gyration_features(x)
    assert g["rg_nm"] > 0 and g["axis_ratio_1_3"] >= 1
    rod = np.c_[np.linspace(0, 10, 200), np.zeros(200), np.zeros(200)] + rng.normal(0, 0.01, (200, 3))
    assert gyration_features(rod)["asphericity"] > 0.9
    t = terminal_coil_lengths(["CCCHHHHEEECC", "HHHHCCCCC"])
    assert t["terminal_coil_total"] == 3 + 2 + 0 + 5 and t["terminal_coil_max"] == 5
    d = max_diameter(rod)
    assert abs(d - np.linalg.norm(rod[0] - rod[-1])) < 0.05
    print("ok  structure features")


# ---------------------------------------------------------------------------
# Synthetic end-to-end calibration
# ---------------------------------------------------------------------------

def write_synthetic_dataset(root: Path, n_prot: int, rng) -> list[str]:
    """
    Proteins of different size. The amplitude each one 'needs' grows with
    size and with a hidden tail length, plus noise, so a size+flexibility
    model should beat a single fixed amplitude.
    """
    pdbs = []
    grid = np.round(np.arange(0, 2.0 + 1e-9, 0.02), 6)
    for i in range(n_prot):
        pdb = f"s{i:03d}"
        n = int(rng.integers(150, 900))
        x = synthetic_protein(n, rng)
        modes, eig = synthetic_modes(x, 10, rng)
        tail = int(rng.integers(0, 25))
        env = {f"{s}_K10": envelope_extents(x, modes, grid, eig, s) for s in ("uniform", "thermal")}
        true_A = 0.05 * (n / 300) ** 0.5 * (1 + tail / 10) * np.exp(rng.normal(0, 0.15))
        req = np.array([envelope_at(grid, env["thermal_K10"][:, a], true_A) for a in range(3)])
        req_body = req * 0.97

        calib = root / pdb / "calib"
        calib.mkdir(parents=True)
        np.savez_compressed(calib / "envelope.npz", grid=grid, n_modes=np.array([10]), **env)
        st = lambda r: {"max_nm": r.tolist(), "p999_nm": (r * 0.99).tolist(),
                        "p99_nm": (r * 0.98).tolist(), "median_nm": (r * 0.9).tolist(), "n_frames": 1000}
        (calib / "required_box.json").write_text(json.dumps({
            "pdb": pdb, "md_box_lengths_nm": (req + 2.4).tolist(),
            "pooled": {"lab": st(req), "body": st(req_body)},
        }))
        e0 = extents(x)
        feats = {"pdb": pdb, "n_heavy_atoms": n, "n_residues": n // 8, "n_chains": 1,
                 **gyration_features(x), "max_diameter_nm": max_diameter(x),
                 "extent0_x_nm": e0[0], "extent0_y_nm": e0[1], "extent0_z_nm": e0[2],
                 "extent0_max_nm": e0.max(),
                 "terminal_coil_total": tail, "terminal_coil_max": tail, "coil_fraction": 0.3}
        (calib / "features.json").write_text(json.dumps(feats))
        (calib / "nma_features.json").write_text(json.dumps({
            "nma_msf_10": float(np.sum(1 / eig) / n), "inv_lambda7": float(1 / eig[0]),
            "mode7_localisation": 3.0}))
        pdbs.append(pdb)
    return pdbs


def test_end_to_end(rng):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        pdbs = write_synthetic_dataset(root, 80, rng)
        cmd = [sys.executable, str(HERE / "calib_fit_model.py"), "--pdbs", *pdbs,
               "--results", str(root), "--max-features", "2", "--cv-repeats", "3",
               "--candidates", "log_n_heavy", "log_rg", "terminal_coil_max", "asphericity",
               "--tag", "synthetic"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        print(res.stdout[-1500:])
        if res.returncode != 0:
            print(res.stderr)
            raise AssertionError("calib_fit_model.py failed")
        s = json.loads((root / "calibration" / "synthetic" / "summary.json").read_text())
        # the hidden drivers are size and tail length
        assert set(s["selected_features"]) & {"terminal_coil_max", "log_n_heavy", "log_rg"}, \
            s["selected_features"]
        t = s["test"]
        assert t["model"]["median_volume_vs_cubic"] < t["fixed_A"]["median_volume_vs_cubic"]
        for f in ("summary.json", "model.json", "test_predictions.csv",
                  "feature_search_cv.csv", "test_methods_volume.png"):
            assert (root / "calibration" / "synthetic" / f).exists(), f
    print("ok  end-to-end calibration on 80 synthetic proteins")


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    test_make_whole(rng)
    test_superpose(rng)
    test_envelope(rng)
    test_features(rng)
    test_end_to_end(rng)
    print("\nall tests passed")
