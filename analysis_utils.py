# analysis_utils.py
from pathlib import Path
import numpy as np
from io_utils import print_header

def normalize_modes(modes: np.ndarray) -> np.ndarray:
    """Row-wise normalisation of mode vectors, safe for zero vectors."""
    norms = np.linalg.norm(modes, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    return modes / norms

def compute_confusion_matrix(
    nma_modes: np.ndarray,
    pca_modes: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute dot-product similarity matrix between NMA and PCA modes.

    nma_modes : shape (n_modes, 3N)
    pca_modes : shape (n_modes, 3N)

    Returns:
        similarity : shape (n_modes, n_modes)
    """
    print_header("Computing confusion matrix between NMA and PCA modes...")

    # Make sure both have same dimensionality
    D = min(nma_modes.shape[1], pca_modes.shape[1])
    if nma_modes.shape[1] != pca_modes.shape[1]:
        print(
            f"[WARNING] Dimension mismatch: NMA={nma_modes.shape[1]}, "
            f"PCA={pca_modes.shape[1]}. Using D={D}."
        )
        nma_modes = nma_modes[:, :D]
        pca_modes = pca_modes[:, :D]

    print("[CHECK] NMA modes shape:", nma_modes.shape)
    print("[CHECK] PCA modes shape:", pca_modes.shape)

    # Normalise rows
    nma_norm = normalize_modes(nma_modes)
    pca_norm = normalize_modes(pca_modes)

    # Absolute dot product similarity matrix
    similarity = np.abs(np.dot(nma_norm, pca_norm.T))

    print(f"[CONFUSION] Similarity matrix shape: {similarity.shape}")
    print(
        f"[CONFUSION] Max value: {np.max(similarity):.4f}, "
        f"Min value: {np.min(similarity):.4f}"
    )

    return similarity

def rmsip(
    modes_a: np.ndarray,
    modes_b: np.ndarray,
    k: int = 10,
) -> float:
    """
    RMSIP between the first k eigenvectors (subspace similarity).

    modes_a: (n_a, D)
    modes_b: (n_b, D)
    """
    a = normalize_modes(modes_a)
    b = normalize_modes(modes_b)

    # Match dimensionality defensively
    D = min(a.shape[1], b.shape[1])
    if a.shape[1] != b.shape[1]:
        print(
            f"[WARNING] RMSIP dimension mismatch: A={a.shape[1]}, B={b.shape[1]}. Using D={D}."
        )
        a = a[:, :D]
        b = b[:, :D]

    k_eff = min(k, a.shape[0], b.shape[0])
    A = a[:k_eff]
    B = b[:k_eff]

    IP = A @ B.T  # (k_eff, k_eff)
    return float(np.sqrt(np.sum(IP ** 2) / k_eff))

def compute_confusion_matrices_per_replica(
    nma_modes: np.ndarray,
    pca_modes_by_replica: list[np.ndarray],
    k_rmsip: int = 10,
) -> list[dict]:
    """
    Compute NMA vs PCA confusion matrix for each PCA replica separately.

    Returns a list of dicts, each containing:
      - replica: int
      - confusion: np.ndarray (n_nma, n_pca)
      - best_per_nma: np.ndarray (n_nma,)  [max similarity in each NMA row]
      - argbest_per_nma: np.ndarray (n_nma,) [index of best PCA mode per NMA mode]
      - rmsip: float
    """
    results = []
    for r, pca_modes in enumerate(pca_modes_by_replica):
        print_header(f"Replica {r}: NMA vs PCA confusion matrix")

        confusion = compute_confusion_matrix(nma_modes, pca_modes)

        best_per_nma = confusion.max(axis=1)        # (n_nma,)
        argbest_per_nma = confusion.argmax(axis=1)  # (n_nma,)

        r_rmsip = rmsip(nma_modes, pca_modes, k=k_rmsip)

        print(f"[REPLICA {r}] RMSIP (k={min(k_rmsip, nma_modes.shape[0], pca_modes.shape[0])}): {r_rmsip:.3f}")

        results.append({
            "replica": r,
            "confusion": confusion,
            "best_per_nma": best_per_nma,
            "argbest_per_nma": argbest_per_nma,
            "rmsip": r_rmsip,
        })

    return results

def aggregate_best_matches(replica_results: list[dict]) -> dict:
    """
    Aggregate best_per_nma across replicas (mean/std) + RMSIP mean/std.
    """
    best_stack = np.stack([d["best_per_nma"] for d in replica_results], axis=0)  # (n_rep, n_nma)
    rmsips = np.array([d["rmsip"] for d in replica_results], dtype=float)

    best_mean = best_stack.mean(axis=0)
    best_std = best_stack.std(axis=0, ddof=1) if best_stack.shape[0] > 1 else np.zeros_like(best_mean)

    return {
        "best_mean": best_mean,
        "best_std": best_std,
        "rmsip_mean": float(rmsips.mean()),
        "rmsip_std": float(rmsips.std(ddof=1)) if rmsips.size > 1 else 0.0,
    }
    
def compute_nma_subspace_capture(
    confusion: np.ndarray,
    kept_pca_idx,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the subspace coverage of each NMA mode by the selected PCA modes.

    capture[m] = Σ_p (u_m · v_p)^2  (bounded ~[0,1] if PCA modes are orthonormal)
    projection[m] = sqrt(capture[m])
    """
    confusion = np.asarray(confusion)

    kept = list(kept_pca_idx)
    if len(kept) == 0:
        raise ValueError("kept_pca_idx is empty.")

    zero_based = any(k == 0 for k in kept)
    kept_cols = kept if zero_based else [k - 1 for k in kept]

    n_nma, n_pca = confusion.shape
    if min(kept_cols) < 0 or max(kept_cols) >= n_pca:
        raise ValueError(f"kept_pca_idx out of range for n_pca={n_pca}.")

    sub = confusion[:, kept_cols]             # (n_nma, n_kept)
    capture = np.sum(sub ** 2, axis=1)        # (n_nma,)
    projection = np.sqrt(capture)

    return capture, projection

def compute_pca_subspace_capture_by_nma(
    confusion: np.ndarray,      # (n_nma, n_pca)
    kept_nma_idx,               # list/array of NMA row indices (0-based)
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each PCA mode p, how much is captured by selected NMA modes?

    capture_p[p] = Σ_m (u_m · v_p)^2 over selected NMA modes m
    proj[m, p]   = (u_m · v_p)^2 for stacking
    """
    kept_nma_idx = np.asarray(kept_nma_idx, dtype=int)
    proj = confusion[kept_nma_idx, :] ** 2            # (k_nma, n_pca)
    capture = proj.sum(axis=0)                        # (n_pca,)
    return capture, proj

def nma_variance_threshold(eigvals: np.ndarray, n_trivial: int = 6):
    """
    eigvals: array of eigenvalues in mode order (including trivial at start).
    Returns (x_modes, cumfrac) where x_modes is 1-based *non-trivial* mode index.
    """
    ev = np.asarray(eigvals, dtype=float)

    # drop trivial modes (usually 6 for ANM; sometimes 5 depending on model)
    ev = ev[n_trivial:]

    # guard against zeros / negatives
    ev = ev[ev > 0]

    w = 1.0 / ev
    cum = np.cumsum(w) / np.sum(w)
    x = np.arange(1, cum.size + 1)
    return x, cum