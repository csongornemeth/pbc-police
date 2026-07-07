# plot_utils.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from io_utils import print_header


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    out_path: Path,
    pdb_code: str,
) -> None:
    print_header("Plotting confusion matrix heatmap...")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap="viridis", aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Absolute Dot Product Similarity")

    ax.set_title(f"Confusion Matrix: NMA vs PCA Modes for {pdb_code.upper()}")
    ax.set_xlabel("PCA Modes")
    ax.set_ylabel("NMA Modes")

    n_nma, n_pca = confusion_matrix.shape

    ax.set_xticks(np.arange(n_pca))
    ax.set_xticklabels(np.arange(1, n_pca + 1))
    ax.set_yticks(np.arange(n_nma))
    ax.set_yticklabels(np.arange(1, n_nma + 1))

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Numeric labels in each cell
    for i in range(n_nma):
        for j in range(n_pca):
            val = confusion_matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[PLOT] Confusion matrix saved to: {out_path}")

def plot_best_match_barplot(
    best_per_nma: np.ndarray,
    argbest_per_nma: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Bar plot of best PCA match per NMA mode, annotated with the PCA mode index.
    """
    print_header("Plotting best-match bar plot...")

    n_modes = best_per_nma.shape[0]
    x = np.arange(1, n_modes + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, best_per_nma)

    ax.set_xlabel("NMA mode index")
    ax.set_ylabel("Best |dot(NMA, PCA)|")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_xticks(x)

    # Annotate PCA index above each bar
    for i, bar in enumerate(bars):
        pca_idx = argbest_per_nma[i] + 1  # +1 for human-readable indexing
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"PC{pca_idx}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[PLOT] Best-match bar plot saved to: {out_path}")

def plot_pca_variance_thresholds(
    explained_variance_ratio: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    targets=(0.75, 0.80, 0.90, 0.95, 0.99),
    save_plot: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evr = np.asarray(explained_variance_ratio, dtype=float).ravel()
    if evr.size == 0:
        raise ValueError("Empty explained variance ratio array provided.")

    # normalise
    s = evr.sum()
    if s <= 0:
        raise ValueError("Sum of explained variance ratios is <= 0, cannot normalise.")
    evr = evr / s

    cum = np.cumsum(evr)

    targets = tuple(float(t) for t in targets)
    n_pcs = []
    achieved = []
    for t in targets:
        if not (0.0 < t <= 1.0):
            raise ValueError(f"Target {t} is out of valid range (0, 1].")
        k = int(np.searchsorted(cum, t, side="left") + 1)  # 1-based PC count
        k = min(k, evr.size)
        n_pcs.append(k)
        achieved.append(float(cum[k - 1]))

    #plot

    plot_path = None
    if save_plot:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, cum.size + 1), cum)

        plt.xticks(np.arange(1, cum.size + 1))  # 👈 ALL integers
        for t in targets:
            plt.axhline(t, linestyle="--")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA cumulative explained variance")
        plt.ylim(0, 1.01)
        plt.xlim(1, cum.size)
        plt.tight_layout()

        plot_path = out_dir / f"{prefix}_pca_explained_variance.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    return {
        "targets": targets,
        "n_pcs": tuple(n_pcs),
        "achieved": tuple(achieved),
        "cumulative": cum,
        "plot": plot_path,
    }

def plot_nma_pca_stacked_bars(
    confusion: np.ndarray,
    kept_pca_idx,
    nma_start: int = 7,
    nma_end: int | None = None,
    title: str = "NMA vs PCA mode overlap",
    outfile: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Stacked bar chart where the stack height indicates how well the subspace
    spanned by the selected PCA modes describes each NMA mode.
    """

    confusion_m = np.asarray(confusion)
    M, P = confusion_m.shape

    kept = list(kept_pca_idx)
    if len(kept) == 0:
        raise ValueError("kept_pca_idx is empty.")

    # PCA indexing: allow 0-based or 1-based inputs
    zero_based = any(k == 0 for k in kept)
    kept_cols = kept if zero_based else [k - 1 for k in kept]

    if min(kept_cols) < 0 or max(kept_cols) >= P:
        raise ValueError(f"kept_pca_idx out of range for P={P} PCA modes.")

    # NMA range (inputs are 1-based)
    if nma_end is None:
        nma_end = M
    if nma_start < 1 or nma_end > M or nma_start > nma_end:
        raise ValueError(f"Invalid NMA range: {nma_start}..{nma_end} for M={M}")

    nma_rows = np.arange(nma_start - 1, nma_end)   # 0-based rows
    x_labels = np.arange(nma_start, nma_end + 1)   # display as 1-based

    # submatrix: selected NMAs x kept PCAs
    D_sub = confusion_m[np.ix_(nma_rows, kept_cols)]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(x_labels)), 6))
    bottom = np.zeros(len(x_labels), dtype=float)

    for j, col in enumerate(kept_cols):
        heights = D_sub[:, j]
        ax.bar(x_labels, heights, bottom=bottom, label=f"PC{col + 1}")
        bottom += heights

    ax.set_xlabel("NMA mode")
    ax.set_ylabel("Sum of |dot(NMA, PCA)| (stacked by PCA)")
    ax.set_title(title)
    ax.set_xticks(x_labels)
    ax.margins(x=0.01)

    if len(kept_cols) <= 10:
        ax.legend(loc="upper right", frameon=True)
    else:
        ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=8)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)

def plot_nma_pca_subspace_overlap(
    capture: np.ndarray,
    nma_start: int = 7,
    nma_end: int | None = None,
    title: str = "NMA subspace capture by selected PCA modes",
    outfile: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Bar plot of subspace capture per NMA mode.
    """
    capture = np.asarray(capture)

    if nma_end is None:
        nma_end = capture.size

    y = capture[nma_start - 1 : nma_end]
    x = np.arange(nma_start, nma_end + 1)

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(x)), 4))
    ax.bar(x, y)

    y_max = max(1.0, float(np.max(y)) * 1.1)

    ax.set_xlabel("NMA mode")
    ax.set_ylabel("Subspace capture Σ(dot²)")
    ax.set_ylim(0, y_max)
    ax.set_title(title)
    ax.set_xticks(x)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        print(f"[PLOT] NMA subspace capture plot saved to: {outfile}")
    else:
        plt.show()

def plot_pca_nma_overlap_stacked(
    proj: np.ndarray,         # (k_nma, n_pca) = (dot^2)
    nma_labels: list[str],    # len = k_nma, e.g. ["NMA7", ...]
    pc_start: int,
    pc_end: int,
    title: str,
    outfile: str | Path,
) -> None:
    outfile = Path(outfile)

    # Clamp pc_end to available PCs
    n_pca = proj.shape[1]
    pc_end_eff = min(pc_end, n_pca)
    if pc_end_eff < pc_start:
        raise ValueError(f"Invalid PC range: pc_start={pc_start}, pc_end={pc_end} with n_pca={n_pca}")

    pcs = np.arange(pc_start, pc_end_eff + 1)  # 1-based labels
    pc_idx0 = pcs - 1                          # to 0-based

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(pcs), dtype=float)

    # If labels list is longer/shorter than proj rows, handle safely
    k_nma = proj.shape[0]
    if len(nma_labels) != k_nma:
        nma_labels = [f"NMA{i+1}" for i in range(k_nma)]

    for i, lab in enumerate(nma_labels):
        vals = proj[i, pc_idx0]
        ax.bar(pcs, vals, bottom=bottom, label=lab)
        bottom += vals

    ax.set_xlabel("PCA mode")
    ax.set_ylabel("Sum of (dot²) | stacked by NMA")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=1)
    ax.set_ylim(0, max(1.0, float(bottom.max()) * 1.05))

    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_nma_cumulative_variance(
    x: np.ndarray,
    cum: np.ndarray,
    title: str,
    outfile: str | Path,
    targets=(0.75, 0.8, 0.9, 0.95, 0.99),
) -> None:
    outfile = Path(outfile)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, cum)
    for t in targets:
        plt.axhline(t, linestyle="--")

    plt.xlabel("Number of non-trivial NMA modes")
    plt.ylabel("Cumulative fraction of (1/λ)")
    plt.title(title)
    plt.ylim(0, 1.01)

    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)