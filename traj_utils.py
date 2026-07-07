# traj_utils.py

import mdtraj as md
import numpy as np

from io_utils import get_pdb_dir, load_full_topology


def build_protein_heavy_views(pdb_code: str):
    """
    Returns:
        traj_protein_heavy_ref : md.Trajectory
            Protein heavy-only reference trajectory for NMA (already sliced).
        top_xtc_protein : md.Topology
            Topology of traj_protein_heavy_ref (protein-heavy only).
        protein_heavy_idx_local : np.ndarray
            Local indices 0..N-1 within the protein-heavy topology.
        protein_heavy_idx_full : np.ndarray
            Indices of protein heavy atoms on the heavy-only FULL topology (top_xtc_full).
            Use this to slice XTC chunks loaded with top_xtc_full.
        top_xtc_full : md.Topology
            Heavy-only topology containing protein + ligand heavy atoms (no water/ions/H).
    """

    pdb_dir = get_pdb_dir(pdb_code)
    traj_full, top_full = load_full_topology(pdb_dir)

    print(f"[traj_utils] Loaded full topology: {top_full.n_atoms} atoms")

    # 1) Heavy-only (no H), no water, no ions → protein + ligand heavy atoms
    ion_resnames = ["NA", "CL"]
    ion_clause = " or ".join(f"resname {x}" for x in ion_resnames)

    sel_full_heavy = top_full.select(
        f"not water and not element H and not ({ion_clause})"
    )

    traj_heavy_full = traj_full.atom_slice(sel_full_heavy)
    top_xtc_full = traj_heavy_full.topology

    print(f"[traj_utils] Heavy-only full topology: {top_xtc_full.n_atoms} atoms")

    # 2) Protein-only heavy atoms (indices on top_xtc_full)
    protein_heavy_idx_full = top_xtc_full.select("protein")

    traj_protein_heavy_ref = traj_heavy_full.atom_slice(protein_heavy_idx_full)
    top_xtc_protein = traj_protein_heavy_ref.topology

    print(
        f"[traj_utils] Protein heavy-only topology: "
        f"{top_xtc_protein.n_atoms} atoms"
    )

    # Local indices (0..N-1) within protein-heavy topology
    protein_heavy_idx_local = np.arange(top_xtc_protein.n_atoms, dtype=int)

    return (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    )
