import os
import numpy as np
from pathlib import Path
from collections import defaultdict

from traj_utils import build_protein_heavy_views
from nma_bio3d import run_aanma_r_from_traj
from pca_stream import run_incremental_pca_from_chunks
from align_core import compute_alignment_core_aidxs

from analysis_utils import (
    compute_confusion_matrices_per_replica,
    compute_nma_subspace_capture,
    compute_pca_subspace_capture_by_nma,
    nma_variance_threshold,
)

from plot_utils import (
    plot_confusion_matrix,
    plot_best_match_barplot,
    plot_nma_pca_stacked_bars,
    plot_nma_pca_subspace_overlap,
    plot_pca_variance_thresholds,
    plot_nma_cumulative_variance,
    plot_pca_nma_overlap_stacked,
)

from io_utils import get_pdb_dir, collect_xtc_paths, print_header


def group_xtc_paths_by_replica(xtc_paths):
    """Group XTC paths by replica index inferred from directory structure."""
    groups = defaultdict(list)

    for p in xtc_paths:
        parts = list(p.parts)
        rep = None

        if "validation" in parts:
            i = parts.index("validation")
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                rep = int(parts[i + 1])

        if rep is None:
            for part in reversed(parts):
                if part.isdigit():
                    rep = int(part)
                    break

        if rep is None:
            raise RuntimeError(f"Could not infer replica index from path: {p}")

        groups[rep].append(p)

    return {r: sorted(ps) for r, ps in sorted(groups.items())}


def main():
    pdb_code = input("Enter PDB code (4 characters): ").strip().lower()
    chunk_size = int(input("Chunk size for IncrementalPCA [500]: ") or 500)

    save_pca_json = (
        input("Save PCA results to JSON? (y/n) [n]: ").strip().lower() == "y"
    )

    n_modes_keep = 20
    k_stack = 10

    out_root = Path.cwd()
    out_dir = out_root / "results" / pdb_code
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_dir = get_pdb_dir(pdb_code)
    xtc_paths = collect_xtc_paths(pdb_dir)

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,   # 0..N-1 in protein-heavy topology
        protein_heavy_idx_full,    # indices on top_xtc_full (used for slicing XTC chunks)
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    print_header("Starting NMA–PCA comparison pipeline")
    print(f"[MAIN] Protein heavy atoms: {traj_protein_heavy_ref.n_atoms}")

    # ----- 1.5) Alignment core (Pattern A) -----
    core_cache = out_dir / f"{pdb_code}_align_core_aidxs.npy"

    if core_cache.exists():
        core_aidxs = np.load(core_cache)
        print(f"[MAIN] Loaded alignment core: {core_cache} (n={core_aidxs.size})")
    else:
        core_aidxs = compute_alignment_core_aidxs(
            xtc_paths=xtc_paths,
            topology=top_xtc_full,
            atom_indices_full=protein_heavy_idx_full,
            max_frames=2000,
            verbose=True,
        )
        np.save(core_cache, core_aidxs)
        print(f"[MAIN] Saved alignment core: {core_cache} (n={core_aidxs.size})")

    # Optional sanity checks
    assert core_aidxs.max() < traj_protein_heavy_ref.n_atoms
    print(f"[MAIN] Alignment core size: {core_aidxs.size}")
    print(f"[MAIN] Protein-heavy atoms (local): {traj_protein_heavy_ref.n_atoms}")
    print(f"[MAIN] Protein-heavy indices on full topology: {protein_heavy_idx_full.size}")
    print(f"[MAIN] Heavy-only full topology atoms: {top_xtc_full.n_atoms}")

    # ----- 3) NMA -----
    nma_modes, nma_eigvals = run_aanma_r_from_traj(
        traj_protein_heavy_ref,
        n_modes_keep,
    )

    nma_modes *= 0.1  # Å to nm conversion
    print("[MAIN] NMA modes converted from Å to nm.")
    print(f"[MAIN] NMA modes shape: {nma_modes.shape}")
    print(f"[MAIN] NMA eigenvalues shape: {nma_eigvals.shape}")

    # --- NMA cumulative variance plot ---
    x_nma, cum_nma = nma_variance_threshold(nma_eigvals, n_trivial=6)
    nma_cum_path = out_dir / f"{pdb_code}_nma_cumulative_variance.png"
    plot_nma_cumulative_variance(
        x=x_nma,
        cum=cum_nma,
        title=f"{pdb_code.upper()} – NMA cumulative variance (1/λ)",
        outfile=nma_cum_path,
    )
    print(f"[MAIN] Saved NMA cumulative variance plot: {nma_cum_path}")

    # ----- 2) PCA (per replica) -----
    xtc_by_rep = group_xtc_paths_by_replica(xtc_paths)
    print_header(f"Found replicas: {list(xtc_by_rep.keys())}")

    pca_modes_by_replica = []
    rep_order = []

    for rep, rep_xtcs in xtc_by_rep.items():
        print_header(f"Running PCA for replica {rep}")
        print(f"[REPLICA {rep}] #xtc files: {len(rep_xtcs)}")

        pca_json_path = (
            out_dir / f"{pdb_code}_rep{rep}_ipca.json"
            if save_pca_json
            else None
        )

        pca_rep, evr_rep = run_incremental_pca_from_chunks(
            xtc_paths=rep_xtcs,
            topology=top_xtc_full,
            n_components=n_modes_keep,
            chunk_size=chunk_size,
            atom_indices=protein_heavy_idx_full,
            align_indices=core_aidxs,
            save_json_path=pca_json_path,
        )

        plot_pca_variance_thresholds(
            explained_variance_ratio=evr_rep,
            out_dir=out_dir,
            prefix=f"{pdb_code}_rep{rep}",
            targets=(0.75, 0.80, 0.90, 0.95, 0.99),
            save_plot=True,
        )

        print(f"[REPLICA {rep}] PCA modes shape: {pca_rep.shape}")

        pca_modes_by_replica.append(pca_rep)
        rep_order.append(rep)

    # ----- 4) Compare (per replica) -----
    rep_results = compute_confusion_matrices_per_replica(
        nma_modes=nma_modes,
        pca_modes_by_replica=pca_modes_by_replica,
        k_rmsip=10,
    )

    for d in rep_results:
        r_idx = d["replica"]
        rep = rep_order[r_idx]
        confusion = d["confusion"]

        kept_pcs = list(range(1, k_stack + 1))

        capture, proj = compute_nma_subspace_capture(
            confusion=confusion,
            kept_pca_idx=kept_pcs,
        )

        plot_nma_pca_subspace_overlap(
            capture=capture,
            nma_start=7,
            nma_end=n_modes_keep,
            title=f"{pdb_code.upper()} – Replica {rep}: NMA subspace capture by first {k_stack} PCs",
            outfile=out_dir / f"{pdb_code}_rep{rep}_overlap_{k_stack}.png",
        )

        plot_nma_pca_stacked_bars(
            confusion=confusion,
            kept_pca_idx=kept_pcs,
            nma_start=7,
            nma_end=n_modes_keep,
            title=f"{pdb_code.upper()} – Replica {rep}: NMA overlap with first {k_stack} PCs (stacked)",
            outfile=out_dir / f"{pdb_code}_rep{rep}_nma_pca_stacked.png",
        )

        # --- PCA overlap stacked by NMA ---
        nma_start = 7
        k_nma_stack = 10
        kept_nma_idx = list(range(nma_start - 1, nma_start - 1 + k_nma_stack))

        _, proj_p = compute_pca_subspace_capture_by_nma(
            confusion=confusion,
            kept_nma_idx=kept_nma_idx,
        )

        plot_pca_nma_overlap_stacked(
            proj=proj_p,
            nma_labels=[f"NMA{i}" for i in range(nma_start, nma_start + k_nma_stack)],
            pc_start=1,
            pc_end=k_stack,
            title=f"{pdb_code.upper()} – Replica {rep}: PCA overlap with NMA{nma_start}–NMA{nma_start+k_nma_stack-1}",
            outfile=out_dir / f"{pdb_code}_rep{rep}_pca_nma_stacked.png",
        )

        plot_confusion_matrix(
            confusion,
            out_dir / f"{pdb_code}_rep{rep}_confusion_matrix.png",
            pdb_code=f"{pdb_code} rep{rep}",
        )

        plot_best_match_barplot(
            best_per_nma=d["best_per_nma"],
            argbest_per_nma=d["argbest_per_nma"],
            out_path=out_dir / f"{pdb_code}_rep{rep}_bestmatch_barplot.png",
            title=f"{pdb_code.upper()} – Replica {rep}: Best PCA match per NMA mode",
        )

    # ----- 5) Global confusion matrix -----
    confusion_global = np.mean(
        np.stack([d["confusion"] for d in rep_results], axis=0),
        axis=0,
    )

    np.save(out_dir / f"{pdb_code}_global_confusion_matrix.npy", confusion_global)

    plot_confusion_matrix(
        confusion_global,
        out_dir / f"{pdb_code}_global_confusion_matrix.png",
        pdb_code=f"{pdb_code} (global mean)",
    )

    print_header("Pipeline finished successfully")


if __name__ == "__main__":
    main()