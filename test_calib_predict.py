#!/usr/bin/env python3
# test_calib_predict.py
"""
Self-tests for calib_predict.py on synthetic data (no MDTraj, R or database).

  python test_calib_predict.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from calib_geometry import make_whole
from calib_predict import assemble_solute, orient_and_centre
from test_calib import write_synthetic_dataset, synthetic_protein, random_rotation

HERE = Path(__file__).resolve().parent


def test_matches_fit(rng):
    """Boxes from calib_predict.py equal the 'model' boxes calib_fit_model.py tested."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        pdbs = write_synthetic_dataset(root, 80, rng)
        fit = [sys.executable, str(HERE / "calib_fit_model.py"), "--pdbs", *pdbs,
               "--results", str(root), "--cv-repeats", "3", "--tag", "synthetic",
               "--candidates", "log_n_heavy", "log_rg", "terminal_coil_max", "asphericity"]
        res = subprocess.run(fit, capture_output=True, text=True, cwd=HERE)
        assert res.returncode == 0, res.stderr
        mdir = root / "calibration" / "synthetic"
        ref = pd.read_csv(mdir / "test_predictions.csv")
        ref = ref[ref["method"] == "model"].set_index("pdb")

        cmd = [sys.executable, str(HERE / "calib_predict.py"), "--pdbs", *ref.index,
               "--results", str(root), "--model-dir", str(mdir), "--name", "heldout"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        print(res.stdout[-900:])
        assert res.returncode == 0, res.stderr
        got = pd.read_csv(mdir / "predictions" / "heldout.csv").set_index("pdb").loc[ref.index]

        assert np.allclose(got["A_pred_nm"], ref["A_pred"], atol=1e-9)
        assert np.allclose(got["A_needed_nm"], ref["A_star"], atol=1e-9)
        for a in "xyz":
            d = got[f"box_{a}"] - ref[f"L{a}"]
            assert d.min() >= -1e-9 and d.max() <= 1e-3 + 1e-9, (d.min(), d.max())
        # rounding the box up can only turn "too small" into "large enough"
        assert not (ref["safe"] & ~got["safe_retrospective"]).any()
        assert (got["safe_retrospective"] == ref["safe"]).mean() > 0.9
        one = json.loads((root / ref.index[0] / "calib" / "predicted_box.json").read_text())
        assert len(one["box_nm"]) == 3
    print("ok  predicted boxes equal the held-out 'model' boxes of calib_fit_model.py")


def test_placement(rng):
    """Wrapped two-chain solute with hydrogens and a ligand is reassembled and placed."""
    for trial in range(6):
        heavy = synthetic_protein(500, rng)
        n_h = len(heavy)
        split = 260                                    # chain B starts here
        # all-atom solute: each heavy atom followed by one "hydrogen", then a ligand
        hyd = heavy + rng.normal(0, 0.06, heavy.shape)
        prot = np.empty((2 * n_h, 3))
        prot[0::2], prot[1::2] = heavy, hyd
        anchor = heavy[np.argmax(heavy[:, 0])]
        lig = anchor + np.array([0.4, 0, 0]) + np.cumsum(rng.normal(0, 0.08, (12, 3)), axis=0)
        true = np.concatenate([prot, lig]) @ random_rotation(rng) + rng.uniform(-4, 4, 3)
        prot_heavy = np.arange(0, 2 * n_h, 2)
        groups = [np.arange(2 * n_h, 2 * n_h + 12)]

        size = true.max(0) - true.min(0)
        box = np.diag(size + rng.uniform(0.8, 1.5, 3))
        inv = np.linalg.inv(box)
        frac = true @ inv
        wrapped = (frac - np.floor(frac)) @ box

        # what step 1 stores: protein heavy atoms made whole (2 chains)
        nma_ref = make_whole(wrapped[prot_heavy], box, [0, split])
        assert np.abs((nma_ref - nma_ref[0]) - (true[prot_heavy] - true[prot_heavy][0])).max() < 1e-6

        whole = assemble_solute(wrapped, box, prot_heavy, nma_ref, groups)
        d = whole - true
        assert np.allclose(d, d[:1], atol=1e-6), "solute not reassembled as one rigid image"
        assert np.allclose(whole[prot_heavy], nma_ref, atol=1e-6)

        # reference orientation = some rotation of the same structure
        Rref = random_rotation(rng)
        ref = nma_ref @ Rref + rng.uniform(-2, 2, 3)
        new_box = (ref.max(0) - ref.min(0)) + 2.0
        xyz, angle = orient_and_centre(whole, prot_heavy, nma_ref, ref, new_box)
        p = xyz[prot_heavy]
        assert np.allclose(p - p.mean(0), ref - ref.mean(0), atol=1e-6), "wrong orientation"
        assert np.allclose(0.5 * (p.min(0) + p.max(0)), 0.5 * new_box, atol=1e-9), "not centred"
        expect = np.degrees(np.arccos(np.clip((np.trace(Rref) - 1) / 2, -1, 1)))
        assert abs(angle - expect) < 1e-6
        # distances inside the solute are unchanged
        i, j = rng.integers(0, len(true), 200), rng.integers(0, len(true), 200)
        assert np.allclose(np.linalg.norm(xyz[i] - xyz[j], axis=1),
                           np.linalg.norm(true[i] - true[j], axis=1), atol=1e-6)
    print("ok  solute reassembled, oriented and centred (2 chains + hydrogens + ligand)")


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    test_placement(rng)
    test_matches_fit(rng)
    print("\nall tests passed")
