#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import mdtraj as md

from align_core import compute_alignment_core_aidxs  # wherever you placed it
# or from your current module if same file

def compute_com_profile(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    top_xtc_protein: md.Topology,
    stat: str = "p95",
    max_core_frames: int = 2000,
    max_com_frames: int = 20000,
    chunk: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute per-residue CA distance to protein COM after stable-core alignment.

    Returns:
        (Nres,) numpy array in nm
    """

    # 1) Compute alignment core (local indices in protein-heavy space)
    core_aidxs = compute_alignment_core_aidxs(
        xtc_paths=xtc_paths,
        topology=top_xtc_full,
        atom_indices_full=protein_heavy_idx_full,
        max_frames=max_core_frames,
        verbose=verbose,
    )

    if verbose:
        print(f"[COM] Core atoms selected: {core_aidxs.size}")

    ca_idx_local = top_xtc_protein.select("name CA")
    if ca_idx_local.size == 0:
        raise ValueError("No CA atoms found in protein-heavy topology.")

    d_chunks = []
    n_frames_total = 0
    ref = None

    for xtc in xtc_paths:
        for tr in md.iterload(xtc.as_posix(), top=top_xtc_full, chunk=chunk):
            trp = tr.atom_slice(protein_heavy_idx_full)

            if ref is None:
                ref = trp[0]

            # superpose to stable core
            trp.superpose(ref, 0, core_aidxs)

            # protein COM (over all atoms in trp)
            com = md.compute_center_of_mass(trp)

            # CA coords
            ca_xyz = trp.xyz[:, ca_idx_local, :]

            d = np.linalg.norm(ca_xyz - com[:, None, :], axis=2)
            d_chunks.append(d)

            n_frames_total += trp.n_frames
            if n_frames_total >= max_com_frames:
                break
        if n_frames_total >= max_com_frames:
            break

    if not d_chunks:
        raise RuntimeError("No frames processed for COM profile.")

    D = np.concatenate(d_chunks, axis=0)

    if stat == "mean":
        return D.mean(axis=0)
    if stat == "p95":
        return np.percentile(D, 95, axis=0)
    if stat == "max":
        return D.max(axis=0)

    raise ValueError(f"Unknown stat: {stat}")