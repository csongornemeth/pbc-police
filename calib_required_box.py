#!/usr/bin/env python3
# calib_required_box.py
"""
Step 1 of the box calibration: measure, from the MD replicas, how much room
each protein actually needed along x, y and z.

For every frame the protein (heavy atoms) is made whole and its extent
(max - min coordinate) along each box axis is recorded in two frames:

  lab  : the MD box frame as simulated. Includes overall tumbling of the
         protein, so this is what a rectangular box must accommodate.
  body : after superposing the frame onto the reference structure (CA atoms).
         Rotation removed, so only internal flexibility remains. This is what
         NM helpers can in principle predict.

A rectangular box with side L_a avoids periodic-image contact closer than
cutoff c along axis a whenever L_a >= extent_a + c, so
  required box side = max over frames of extent_a + c.
The cutoff is applied later (calib_fit_model.py), so it can be varied.

The reference structure is frame 0 of the first replica. Its coordinates are
saved and reused by calib_nm_envelope.py so both steps share one orientation.

Outputs (results/<pdb>/calib/):
  extents.npz              per-frame extents (lab, body), replica id, time
  required_box.json        summary statistics per replica and pooled
  features.json            structure descriptors of the reference
  reference_xyz.npy        (N, 3) reference coordinates, nm
  nma_reference_xyz.npy    (N, 3) coordinates NMA is computed on (npt.gro)
  reference.pdb            reference structure

Example:
  python calib_required_box.py --pdb 1abc
  python calib_required_box.py --pdb 1abc --stride 5 --chunk 1000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import mdtraj as md

from io_utils import get_pdb_dir, collect_xtc_paths, print_header
from traj_utils import build_protein_heavy_views
from calib_geometry import (
    make_whole,
    superpose,
    extents,
    gyration_features,
    max_diameter,
    terminal_coil_lengths,
)


def chain_starts(top: md.Topology) -> list[int]:
    starts = []
    for ch in top.chains:
        atoms = list(ch.atoms)
        if atoms:
            starts.append(atoms[0].index)
    return sorted(starts)


def group_by_replica(xtc_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in xtc_paths:
        groups[p.parent.name].append(p)

    def key(r):
        return (0, int(r)) if r.isdigit() else (1, r)

    return {r: groups[r] for r in sorted(groups, key=key)}


def stats(ext: np.ndarray) -> dict:
    """ext (F, 3) -> summary per axis."""
    return {
        "max_nm": ext.max(0).tolist(),
        "p999_nm": np.percentile(ext, 99.9, axis=0).tolist(),
        "p99_nm": np.percentile(ext, 99, axis=0).tolist(),
        "median_nm": np.median(ext, axis=0).tolist(),
        "n_frames": int(ext.shape[0]),
    }


def structure_features(ref_xyz: np.ndarray, top: md.Topology) -> dict:
    masses = np.array(
        [a.element.mass if a.element is not None else 12.0 for a in top.atoms]
    )
    feats = {
        "n_heavy_atoms": int(top.n_atoms),
        "n_residues": int(top.n_residues),
        "n_chains": int(sum(1 for c in top.chains if c.n_atoms > 0)),
        **gyration_features(ref_xyz, masses),
        "max_diameter_nm": max_diameter(ref_xyz),
    }
    ext0 = extents(ref_xyz)
    feats.update({
        "extent0_x_nm": float(ext0[0]),
        "extent0_y_nm": float(ext0[1]),
        "extent0_z_nm": float(ext0[2]),
        "extent0_max_nm": float(ext0.max()),
    })

    try:
        tr = md.Trajectory(ref_xyz[None].astype(np.float32), top)
        ss = md.compute_dssp(tr, simplified=True)[0]
        per_chain = defaultdict(list)
        for res, s in zip(top.residues, ss):
            if s != "NA":
                per_chain[res.chain.index].append(s)
        feats.update(terminal_coil_lengths(["".join(v) for v in per_chain.values()]))
    except Exception as exc:  # DSSP needs N, CA, C, O; skip gracefully
        print(f"[WARN] DSSP failed, coil features skipped: {exc}")

    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--outdir", default="results", type=Path,
                    help="Root output dir; files go to <outdir>/<pdb>/calib")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1,
                    help="Use every n-th frame. 1 = all frames (safest for max).")
    ap.add_argument("--align-selection", default="name CA",
                    help="Atoms for body-frame superposition (MDTraj syntax).")
    args = ap.parse_args()

    pdb = args.pdb.strip().lower()
    out = args.outdir / pdb / "calib"
    out.mkdir(parents=True, exist_ok=True)

    pdb_dir = get_pdb_dir(pdb)
    xtcs = group_by_replica(collect_xtc_paths(pdb_dir))

    (
        traj_protein_heavy_ref,   # npt.gro, protein heavy atoms, NMA structure
        top_prot,
        _idx_local,
        prot_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb)

    starts = chain_starts(top_prot)
    align_idx = top_prot.select(args.align_selection)
    if align_idx.size < 3:
        raise ValueError(f"Alignment selection '{args.align_selection}' gave {align_idx.size} atoms")

    # --- NMA reference (npt.gro): make sure it is whole ----------------------
    nma_xyz = traj_protein_heavy_ref.xyz[0].astype(np.float64)
    if traj_protein_heavy_ref.unitcell_vectors is not None:
        nma_whole = make_whole(nma_xyz, traj_protein_heavy_ref.unitcell_vectors[0], starts)
        shift = np.abs(nma_whole - nma_xyz).max()
        if shift > 0.05:
            print(f"[WARN] npt.gro protein was broken across PBC (max shift {shift:.2f} nm). "
                  "NMA must be computed on a whole structure; the whole version is saved.")
        nma_xyz = nma_whole
    np.save(out / "nma_reference_xyz.npy", nma_xyz)

    print_header(f"Required box from MD: {pdb}")
    print(f"Replicas: {list(xtcs)} | atoms: {top_prot.n_atoms} | chains: {len(starts)}")

    ref = None
    ext_lab, ext_body, rep_ids, times = [], [], [], []
    box0 = None
    per_rep = {}

    for rep, paths in xtcs.items():
        lab_r, body_r = [], []
        for xtc in paths:
            for tr in md.iterload(xtc.as_posix(), top=top_xtc_full,
                                  chunk=args.chunk, stride=args.stride):
                xyz = tr.xyz[:, prot_idx_full, :]
                box = tr.unitcell_vectors
                if box is None:
                    raise RuntimeError(f"No box in {xtc}; cannot unwrap PBC.")
                whole = make_whole(xyz, box, starts)
                if ref is None:
                    ref = whole[0].copy()
                    box0 = tr.unitcell_lengths[0].astype(float)
                lab = extents(whole)
                body = extents(superpose(whole, ref, align_idx))
                lab_r.append(lab)
                body_r.append(body)
                rep_ids.append(np.full(len(lab), rep, dtype=object))
                times.append(tr.time)
        if not lab_r:
            print(f"[WARN] replica {rep}: no frames")
            continue
        lab_r = np.concatenate(lab_r)
        body_r = np.concatenate(body_r)
        ext_lab.append(lab_r)
        ext_body.append(body_r)
        per_rep[rep] = {"lab": stats(lab_r), "body": stats(body_r)}
        print(f"[rep {rep}] frames={len(lab_r)}  "
              f"max lab={np.round(lab_r.max(0), 2)}  max body={np.round(body_r.max(0), 2)}")

    ext_lab = np.concatenate(ext_lab)
    ext_body = np.concatenate(ext_body)

    np.savez_compressed(
        out / "extents.npz",
        lab=ext_lab.astype(np.float32),
        body=ext_body.astype(np.float32),
        replica=np.concatenate(rep_ids).astype(str),
        time_ps=np.concatenate(times).astype(np.float32),
    )
    np.save(out / "reference_xyz.npy", ref)
    md.Trajectory(ref[None].astype(np.float32), top_prot).save_pdb(
        (out / "reference.pdb").as_posix()
    )

    summary = {
        "pdb": pdb,
        "stride": args.stride,
        "align_selection": args.align_selection,
        "md_box_lengths_nm": box0.tolist(),
        # Smallest (L_a - extent_a) seen: a quick upper bound on how close
        # periodic images came along each axis in the simulated box.
        "min_axis_margin_nm": (box0 - ext_lab.max(0)).tolist(),
        "pooled": {"lab": stats(ext_lab), "body": stats(ext_body)},
        "per_replica": per_rep,
    }
    (out / "required_box.json").write_text(json.dumps(summary, indent=2))

    feats = structure_features(ref, top_prot)
    feats["pdb"] = pdb
    (out / "features.json").write_text(json.dumps(feats, indent=2))

    print_header("Done")
    print(f"pooled max extent lab : {np.round(ext_lab.max(0), 3)} nm")
    print(f"pooled max extent body: {np.round(ext_body.max(0), 3)} nm")
    print(f"MD box                : {np.round(box0, 3)} nm")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
