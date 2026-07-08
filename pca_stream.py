from __future__ import annotations

from pathlib import Path
import json
import gzip
import numpy as np
import mdtraj as md
from sklearn.decomposition import IncrementalPCA

from io_utils import print_header


def yield_pca_chunks(
    xtc_paths: list[Path],
    topology: md.Topology,
    chunk_size: int,
    atom_indices: np.ndarray,
    align_indices: np.ndarray | None = None,
):
    print_header("Streaming trajectory chunks for IncrementalPCA")

    ref_coords = None
    ref_topology = None
    ref_n_atoms = None

    for xtc in xtc_paths:
        print(f"[CHUNK] Reading from trajectory file: {xtc}")

        for traj_chunk in md.iterload(
            xtc.as_posix(),
            top=topology,
            chunk=chunk_size,
        ):
            print(f"[CHUNK] traj_chunk.n_atoms before slice = {traj_chunk.n_atoms}")
            print(f"[CHUNK] len(atom_indices) = {len(atom_indices)}")

            traj_sel = traj_chunk.atom_slice(atom_indices)

            print(f"[CHUNK] traj_sel.n_atoms after slice = {traj_sel.n_atoms}")

            if traj_sel.n_atoms != len(atom_indices):
                raise ValueError(
                    f"[ERROR] Sliced atom count mismatch in file {xtc}: "
                    f"len(atom_indices)={len(atom_indices)}, "
                    f"traj_sel.n_atoms={traj_sel.n_atoms}"
                )

            if ref_coords is None:
                ref_coords = traj_sel[0].xyz.copy()
                ref_topology = traj_sel.topology
                ref_n_atoms = traj_sel.n_atoms
                print(f"[CHUNK] Global reference frame set with {traj_sel.n_atoms} atoms")
            else:
                if traj_sel.n_atoms != ref_n_atoms:
                    raise ValueError(
                        f"[ERROR] Atom count mismatch across chunks/files.\n"
                        f"Reference atoms: {ref_n_atoms}\n"
                        f"Current chunk atoms: {traj_sel.n_atoms}\n"
                        f"File: {xtc}"
                    )

            ref_traj = md.Trajectory(ref_coords.copy(), ref_topology)

            if align_indices is not None:
                if np.max(align_indices) >= traj_sel.n_atoms:
                    raise ValueError(
                        f"[ERROR] align_indices out of bounds for file {xtc}: "
                        f"max(align_indices)={np.max(align_indices)}, "
                        f"traj_sel.n_atoms={traj_sel.n_atoms}"
                    )

            if align_indices is None:
                traj_sel.superpose(ref_traj)
            else:
                traj_sel.superpose(ref_traj, atom_indices=align_indices)

            xyz = traj_sel.xyz
            n_frames_chunk, n_atoms_sel, _ = xyz.shape
            X_chunk = xyz.reshape(n_frames_chunk, n_atoms_sel * 3)

            print(f"[CHUNK] Yielding chunk with shape: {X_chunk.shape}")
            yield X_chunk


def run_incremental_pca_from_chunks(
    xtc_paths: list[Path],
    topology: md.Topology,
    n_components: int,
    chunk_size: int,
    atom_indices: np.ndarray,
    align_indices: np.ndarray | None = None,
    save_json_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream XTCs, align frames, and fit IncrementalPCA.
    """
    print_header("Running IncrementalPCA from streamed chunks")

    ipca = None
    total_frames = 0
    n_features = None

    for X_chunk in yield_pca_chunks(
        xtc_paths=xtc_paths,
        topology=topology,
        chunk_size=chunk_size,
        atom_indices=atom_indices,
        align_indices=align_indices,
    ):
        n_frames_chunk, n_features_chunk = X_chunk.shape
        total_frames += n_frames_chunk

        if ipca is None:
            n_features = n_features_chunk
            n_components_eff = min(n_components, n_features)
            print(f"[IPCA] First chunk shape: {X_chunk.shape}")
            print(f"[IPCA] Using n_components = {n_components_eff}")
            ipca = IncrementalPCA(n_components=n_components_eff)

        if n_features_chunk != n_features:
            raise ValueError(
                f"Chunk feature mismatch: expected {n_features}, got {n_features_chunk}"
            )

        ipca.partial_fit(X_chunk)

    if ipca is None:
        raise RuntimeError("No data chunks were produced. Check xtc_paths and atom_indices.")

    print(f"[IPCA] Total frames processed: {total_frames}")
    print(f"[IPCA] Final components shape: {ipca.components_.shape}")
    print(
        "[IPCA] Explained variance ratio (first few):",
        ipca.explained_variance_ratio_[: min(5, ipca.n_components_)],
    )

    if save_json_path is not None:
        save_incremental_pca_to_json(ipca, save_json_path)

    return ipca.components_, ipca.explained_variance_ratio_


def save_incremental_pca_to_json(ipca, out_path: Path):
    """
    Save a fitted sklearn IncrementalPCA object to a compressed JSON (.json.gz).
    """
    out_path = Path(out_path)

    if out_path.suffix != ".gz":
        out_path = out_path.with_suffix(out_path.suffix + ".gz")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    jso = {
        "components_": ipca.components_.tolist(),
        "explained_variance_": ipca.explained_variance_.tolist(),
        "explained_variance_ratio_": ipca.explained_variance_ratio_.tolist(),
        "singular_values_": ipca.singular_values_.tolist(),
        "mean_": ipca.mean_.tolist(),
        "n_components_": int(ipca.n_components_),
        "noise_variance_": (
            ipca.noise_variance_.tolist()
            if np.ndim(ipca.noise_variance_) > 0
            else float(ipca.noise_variance_)
        ),
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as fp:
        json.dump(jso, fp)

    print(f"[IPCA] Saved compressed PCA JSON to: {out_path}")