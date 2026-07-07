from __future__ import annotations

from pathlib import Path
import numpy as np
import mdtraj as md


def sample_protein_heavy_frames(
    xtc_paths: list[Path],
    topology: md.Topology,
    atom_indices_full: np.ndarray,
    max_frames: int = 2000,
    chunk: int = 200,
) -> md.Trajectory:
    """
    Load up to max_frames frames (protein-heavy sliced) across xtc_paths.
    Returns a trajectory in the *sliced* index space (0..n_atoms_sel-1).
    """
    pieces = []
    n = 0

    for xtc in xtc_paths:
        for tr in md.iterload(xtc.as_posix(), top=topology, chunk=chunk):
            tr_sel = tr.atom_slice(atom_indices_full)
            pieces.append(tr_sel)
            n += tr_sel.n_frames
            if n >= max_frames:
                break
        if n >= max_frames:
            break

    if not pieces:
        raise RuntimeError("No frames sampled. Check xtc_paths/topology/atom_indices_full.")

    traj = md.join(pieces, check_topology=True)
    if traj.n_frames > max_frames:
        traj = traj[:max_frames]

    return traj


def superposeTRAJ_optimised_core(
    traj: md.Trajectory,
    initial_aidxs: np.ndarray | None = None,
    exclusive: bool = False,
    max_iter: int = 25,
    min_frac: float = 1.0 / 3.0,
    cutoff0: float = 0.20,
    cutoff_step: float = 0.01,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively refine alignment atom indices to a stable, low-RMSF core.
    Returns (core_aidxs, rmsf_per_atom).

    Notes:
    - cutoff in nm (MDTraj uses nm).
    - core_aidxs are indices in traj's topology (0..traj.n_atoms-1).
    """
    ref = traj[0]

    if initial_aidxs is None:
        aidxs = np.arange(ref.top.n_atoms, dtype=int)
        exclusive = False
    else:
        aidxs = np.array(initial_aidxs, dtype=int)

    for _ in range(max_iter):
        traj.superpose(ref, 0, aidxs)

        # average coordinates over frames
        avg = traj.xyz.mean(axis=0)  # (n_atoms, 3)
        ref.xyz[0] = avg.copy()

        # RMSF-like metric about avg structure (nm)
        dif = traj.xyz - avg[None, :, :]
        rmsf = np.sqrt(np.sum(dif * dif, axis=(0, 2)) / traj.n_frames)

        cutoff = cutoff0
        while True:
            tmp = np.where(rmsf < cutoff)[0]
            if tmp.size > ref.top.n_atoms * min_frac:
                break
            cutoff += cutoff_step

        if exclusive and initial_aidxs is not None:
            tmp = np.intersect1d(tmp, initial_aidxs)

        if verbose:
            print(
                f"[core] keep={tmp.size}/{rmsf.size} cutoff={cutoff:.2f}nm "
                f"mean_rmsf={rmsf.mean():.3f} mean_rmsf(core)={rmsf[tmp].mean():.3f}"
            )

        if np.array_equal(aidxs, tmp):
            return tmp, rmsf

        aidxs = tmp

    return aidxs, rmsf


def compute_alignment_core_aidxs(
    xtc_paths: list[Path],
    topology: md.Topology,
    atom_indices_full: np.ndarray,
    max_frames: int = 2000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Pattern A: sample frames once, compute a fixed core, return core indices in
    the sliced protein-heavy topology (0..n_atoms_sel-1).
    """
    traj_sample = sample_protein_heavy_frames(
        xtc_paths=xtc_paths,
        topology=topology,
        atom_indices_full=atom_indices_full,
        max_frames=max_frames,
    )

    # Start from all protein-heavy atoms (local index space)
    init = np.arange(traj_sample.n_atoms, dtype=int)

    core_aidxs, _ = superposeTRAJ_optimised_core(
        traj=traj_sample,
        initial_aidxs=init,
        exclusive=False,
        verbose=verbose,
    )
    return core_aidxs
