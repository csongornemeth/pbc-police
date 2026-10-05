#!/usr/bin/env python3
# calib_predict.py
"""
Step 4 of the box calibration: apply a fitted amplitude rule to proteins and
build their boxes.

For every protein:
    descriptors (features.json, nma_features.json)
      -> amplitude A from the rule in <model-dir>/model.json
      -> box side L_a = NM envelope extent along a at amplitude A + cutoff

The box is defined exactly as in calib_fit_model.py (same envelope, same
scaling, K and cutoff, all read from <model-dir>/summary.json), so the
coverage measured there is the coverage of the boxes written here.

Needs steps 1 and 2 (calib_required_box.py, calib_nm_envelope.py) for each
protein. Because step 1 reads the existing MD, the predicted box is also
checked against that MD ("retrospective check"): would the new box have held
everything the old simulation explored? For proteins whose old box was too
small this is the check to pass before rerunning them.

Outputs
  results/<pdb>/calib/predicted_box.json           per protein
  results/<pdb>/calib/predicted_box.gro            with --write-gro
  <model-dir>/predictions/<name>.csv               one row per protein

--write-gro writes the solute (protein + ligand, with hydrogens) taken from
build/npt.gro, made whole, rotated into the orientation the box was computed
in, and centred in the new box. It still has to be solvated, neutralised and
equilibrated. The box is only valid in this orientation.

Examples
  python calib_predict.py --model-dir results/calibration/lab_max_thermal_K10_q0.95 \
      --pdb-list failed_pdbs.txt
  python calib_predict.py --model-dir results/calibration/lab_max_thermal_K10_q0.95 \
      --pdbs 1abc 2xyz --write-gro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calib_geometry import envelope_at, solve_amplitude, make_whole, kabsch_rotations

ION_RESNAMES = ("NA", "CL")          # same as traj_utils.build_protein_heavy_views


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

def read_pdb_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split()[0].lower())
    return out


def feature_vector(model: dict, feats: dict) -> np.ndarray:
    """Descriptor values in the order and transform the rule was fitted with."""
    x = []
    for name in model["features"]:
        src, tf = model["feature_definitions"][name]
        v = feats.get(src)
        v = np.nan if v is None else float(v)
        if tf == "log":
            v = np.log(v) if v > 0 else np.nan
        x.append(v)
    return np.asarray(x, dtype=float)


def predict_amplitude(model: dict, x: np.ndarray) -> float:
    """A = max(exp(intercept + shift + sum coef (x - mean) / sd) - EPS, 0), in nm."""
    y = model["intercept"] + model["conformal_shift"]
    if len(x):
        z = (x - np.asarray(model["standardisation_mean"])) / np.asarray(model["standardisation_sd"])
        y += float(z @ np.asarray(model["coef_standardised"]))
    return float(max(np.exp(y) - model["EPS"], 0.0))


def predict_one(calib: Path, model: dict, settings: dict, train_range: dict,
                baseline_growth: float) -> dict:
    req = json.loads((calib / "required_box.json").read_text())
    env = np.load(calib / "envelope.npz")
    feats = json.loads((calib / "features.json").read_text())
    feats.update(json.loads((calib / "nma_features.json").read_text()))

    key = f"{settings['scaling']}_K{settings['n_modes']}"
    if key not in env:
        raise KeyError(f"envelope.npz has no '{key}'")
    grid = env["grid"].astype(float)
    curve = env[key].astype(float)
    cutoff = float(settings["cutoff"])

    x = feature_vector(model, feats)
    if not np.all(np.isfinite(x)):
        missing = [n for n, v in zip(model["features"], x) if not np.isfinite(v)]
        raise ValueError(f"descriptor(s) missing: {missing}")
    A = predict_amplitude(model, x)
    ext = np.array([envelope_at(grid, curve[:, a], A) for a in range(3)])
    box = np.ceil((ext + cutoff) * 1000.0 - 1e-6) / 1000.0     # .gro precision, never rounded down

    # Descriptors outside what the rule was fitted on: the rule is extrapolating.
    outside = [n for n, v in zip(model["features"], x)
               if n in train_range and not (train_range[n][0] <= v <= train_range[n][1])]

    # Retrospective check against the existing MD of this protein.
    need_ext = np.asarray(req["pooled"][settings["frame"]][f"{settings['stat']}_nm"], dtype=float)
    need = need_ext + cutoff
    a_star = max(solve_amplitude(grid, curve[:, a], need_ext[a]) for a in range(3))
    md_box = np.asarray(req["md_box_lengths_nm"], dtype=float)

    # Baselines on the same footing.
    ext0 = np.array([feats["extent0_x_nm"], feats["extent0_y_nm"], feats["extent0_z_nm"]])
    cubic = (feats["max_diameter_nm"] + 2 * float(settings["standard_d"])) ** 3
    pad_box = ext0 + baseline_growth + cutoff

    return {
        "A_pred_nm": A,
        "A_needed_nm": float(a_star),
        "amplitude_beyond_grid": bool(A > grid[-1]),
        "features_outside_training": outside,
        "box_nm": box.tolist(),
        "volume_nm3": float(np.prod(box)),
        "md_box_nm": md_box.tolist(),
        "volume_vs_md_box": float(np.prod(box) / np.prod(md_box)),
        "volume_vs_cubic": float(np.prod(box) / cubic),
        "needed_box_nm": need.tolist(),
        "safe_retrospective": bool(np.all(box >= need - 1e-9)),
        "worst_shortfall_nm": float(np.max(need - box)),
        "md_box_worst_shortfall_nm": float(np.max(need - md_box)),
        "padded_box_nm": pad_box.tolist(),
        "padded_safe_retrospective": bool(np.all(pad_box >= need - 1e-9)),
        "padded_volume_vs_md_box": float(np.prod(pad_box) / np.prod(md_box)),
    }


# ---------------------------------------------------------------------------
# Placing the solute in the new box
# ---------------------------------------------------------------------------

def assemble_solute(sol_xyz: np.ndarray, box: np.ndarray, prot_heavy: np.ndarray,
                    nma_ref: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """
    Make the all-atom solute whole, consistently with step 1.

    sol_xyz    : (M, 3) solute atoms as in npt.gro (possibly wrapped)
    box        : (3, 3) box vectors of npt.gro
    prot_heavy : indices (into the solute) of the protein heavy atoms
    nma_ref    : (N, 3) the same protein heavy atoms as made whole by step 1
                 (nma_reference_xyz.npy)
    groups     : index arrays of the non-protein residues (ligands, cofactors)

    Protein atoms follow the lattice image step 1 chose for their chain;
    every other residue takes the periodic image nearest to the protein.
    """
    walked = make_whole(sol_xyz, box)
    inv = np.linalg.inv(box)

    delta = nma_ref - walked[prot_heavy]
    cells = np.round(delta @ inv)
    resid = np.abs(delta - cells @ box).max()
    if resid > 0.01:
        raise ValueError(f"solute does not match nma_reference_xyz.npy (residual {resid:.3f} nm); "
                         "different npt.gro or atom selection")
    owner = np.clip(np.searchsorted(prot_heavy, np.arange(len(walked)), side="right") - 1, 0, None)
    whole = walked + (cells @ box)[owner]

    lattice = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) @ box
    centre = nma_ref.mean(0)
    for g in groups:
        x = walked[g]
        x = x - np.round((x.mean(0) - centre) @ inv) @ box
        best, best_d = x, np.inf
        for s in lattice:
            c = x.mean(0) + s
            d = np.min(((nma_ref - c) ** 2).sum(1))
            if d < best_d:
                best, best_d = x + s, d
        whole[g] = best
    return whole


def orient_and_centre(whole: np.ndarray, prot_heavy: np.ndarray, nma_ref: np.ndarray,
                      ref: np.ndarray, box_lengths: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Rotate into the reference orientation (the frame the box was computed in,
    same rotation as calib_nm_envelope.py) and put the protein's bounding box
    in the middle of the new box. Returns coordinates and the rotation angle
    in degrees.
    """
    R, c_mob, _ = kabsch_rotations(nma_ref[None], ref)
    out = (whole - c_mob[0]) @ R[0]
    p = out[prot_heavy]
    out += 0.5 * np.asarray(box_lengths) - 0.5 * (p.min(0) + p.max(0))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R[0]) - 1.0) / 2.0, -1.0, 1.0))))
    return out, angle


def write_gro(pdb: str, calib: Path, box_lengths: np.ndarray, cutoff: float) -> dict:
    import mdtraj as md
    from io_utils import get_pdb_dir, get_topology_path

    full = md.load(get_topology_path(get_pdb_dir(pdb)).as_posix())
    ions = " or ".join(f"resname {r}" for r in ION_RESNAMES)
    sol = full.atom_slice(full.topology.select(f"not water and not ({ions})"))
    top = sol.topology

    prot_heavy = top.select("protein and not element H")
    nma_ref = np.load(calib / "nma_reference_xyz.npy")
    ref = np.load(calib / "reference_xyz.npy")
    if len(prot_heavy) != len(nma_ref):
        raise ValueError(f"{len(prot_heavy)} protein heavy atoms in npt.gro, "
                         f"{len(nma_ref)} in nma_reference_xyz.npy")

    prot_all = set(top.select("protein").tolist())
    groups = []
    for res in top.residues:
        idx = np.array([a.index for a in res.atoms if a.index not in prot_all], dtype=int)
        if idx.size:
            groups.append(idx)

    whole = assemble_solute(sol.xyz[0].astype(np.float64), full.unitcell_vectors[0].astype(np.float64),
                            prot_heavy, nma_ref, groups)
    xyz, angle = orient_and_centre(whole, prot_heavy, nma_ref, ref, box_lengths)

    out = md.Trajectory(xyz[None].astype(np.float32), top)
    out.unitcell_vectors = np.diag(box_lengths)[None].astype(np.float32)
    path = calib / "predicted_box.gro"
    out.save_gro(path.as_posix())

    # Hydrogens and ligands are not part of the calibration: report how far
    # they reach beyond the protein heavy atoms the box was sized on.
    p = xyz[prot_heavy]
    overhang = np.maximum(p.min(0) - xyz.min(0), xyz.max(0) - p.max(0))
    return {
        "gro": path.as_posix(),
        "n_solute_atoms": int(top.n_atoms),
        "n_non_protein_residues": len(groups),
        "rotation_from_npt_gro_deg": angle,
        "all_atom_overhang_nm": overhang.tolist(),
        "start_margin_nm": (np.asarray(box_lengths) - (xyz.max(0) - xyz.min(0)) - cutoff).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-list", type=Path, help="Text file, one PDB code per line")
    src.add_argument("--pdbs", nargs="+")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Folder written by calib_fit_model.py (model.json, summary.json)")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--baseline-growth", type=float, default=1.1,
                    help="Growth allowed by the constant-padding baseline (nm): "
                         "L = starting extent + growth + cutoff")
    ap.add_argument("--write-gro", action="store_true",
                    help="Write the solute in the predicted box (needs MDTraj and the database)")
    ap.add_argument("--name", default=None, help="Name of the output table (default: list file name)")
    args = ap.parse_args()

    model = json.loads((args.model_dir / "model.json").read_text())
    settings = json.loads((args.model_dir / "summary.json").read_text())["settings"]
    settings["n_modes"] = int(settings["n_modes"])

    train_range = {}
    ds = args.model_dir / "dataset.csv"
    if ds.exists():
        d = pd.read_csv(ds)
        train_range = {f: (float(d[f].min()), float(d[f].max())) for f in model["features"] if f in d}

    pdbs = read_pdb_list(args.pdb_list) if args.pdb_list else [p.lower() for p in args.pdbs]
    name = args.name or (args.pdb_list.stem if args.pdb_list else "selection")

    print(f"rule     : {' + '.join(model['features']) or '(one amplitude for all)'}  "
          f"(quantile {model['quantile']})")
    print(f"envelope : {settings['scaling']} K={settings['n_modes']}  frame={settings['frame']}  "
          f"stat={settings['stat']}  cutoff={settings['cutoff']} nm\n")

    rows, skipped = [], []
    for pdb in pdbs:
        calib = args.results / pdb / "calib"
        try:
            r = predict_one(calib, model, settings, train_range, args.baseline_growth)
            if args.write_gro:
                r["placement"] = write_gro(pdb, calib, np.asarray(r["box_nm"]), float(settings["cutoff"]))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[skip] {pdb}: {exc}")
            skipped.append(pdb)
            continue
        r = {"pdb": pdb, "model_dir": args.model_dir.as_posix(), **r}
        (calib / "predicted_box.json").write_text(json.dumps(r, indent=2))
        rows.append(r)

    if not rows:
        raise SystemExit("no protein could be predicted")

    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        for k in ("box_nm", "md_box_nm", "needed_box_nm"):
            f.update({f"{k[:-3]}_{a}": r[k][i] for i, a in enumerate("xyz")})
        f["features_outside_training"] = "+".join(r["features_outside_training"])
        if "placement" in r:
            f["rotation_from_npt_gro_deg"] = r["placement"]["rotation_from_npt_gro_deg"]
            f["max_all_atom_overhang_nm"] = max(r["placement"]["all_atom_overhang_nm"])
        flat.append(f)
    df = pd.DataFrame(flat)
    outdir = args.model_dir / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)

    n = len(df)
    print(f"\n{n} proteins predicted, {len(skipped)} skipped")
    print(f"{'box':18s} {'held old MD':>12s} {'median vol / MD box':>20s} {'worst short (nm)':>17s}")
    print(f"{'NM rule':18s} {df['safe_retrospective'].mean():12.1%} "
          f"{df['volume_vs_md_box'].median():20.2f} {df['worst_shortfall_nm'].max():17.3f}")
    print(f"{'constant padding':18s} {df['padded_safe_retrospective'].mean():12.1%} "
          f"{df['padded_volume_vs_md_box'].median():20.2f} {'':>17s}")
    print(f"{'MD box as run':18s} {(df['md_box_worst_shortfall_nm'] <= 1e-9).mean():12.1%} "
          f"{1.0:20.2f} {df['md_box_worst_shortfall_nm'].max():17.3f}")

    flagged = df[(df["features_outside_training"] != "") | df["amplitude_beyond_grid"]]
    if len(flagged):
        print(f"\n[warn] {len(flagged)} proteins are outside what the rule was fitted on "
              f"(descriptor range or amplitude grid): {flagged['pdb'].tolist()}")
    bad = df[~df["safe_retrospective"]]
    if len(bad):
        print(f"[warn] predicted box would not have held the old MD for: {bad['pdb'].tolist()}")
    if "rotation_from_npt_gro_deg" in df:
        print(f"rotation npt.gro -> box orientation: median "
              f"{df['rotation_from_npt_gro_deg'].median():.1f} deg, max "
              f"{df['rotation_from_npt_gro_deg'].max():.1f} deg")
        print(f"hydrogens/ligands beyond the protein heavy atoms: max "
              f"{df['max_all_atom_overhang_nm'].max():.2f} nm")
    print(f"\nwritten to {outdir / (name + '.csv')}")


if __name__ == "__main__":
    main()
