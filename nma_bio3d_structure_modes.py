# nma_bio3d_structure_modes.py

import numpy as np
import mdtraj as md

from pathlib import Path


from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

from io_utils import print_header


def traj_to_pdb_string(traj: md.Trajectory) -> str:
    """
    Convert an MDTraj trajectory to a PDB string for Bio3D.
    MDTraj coordinates are in nm; PDB coordinates are written in Å.
    """
    xyz = traj.xyz[0]

    lines = []
    serial = 1

    for atom, (x, y, z) in zip(traj.topology.atoms, xyz):
        res = atom.residue

        resname = (res.name or "RES")[:3]
        atomname = (atom.name or "X")[:4]
        element = atom.element.symbol if atom.element is not None else ""

        xA, yA, zA = x * 10.0, y * 10.0, z * 10.0
        resseq = res.index + 1

        line = (
            f"ATOM  {serial:5d} "
            f"{atomname:<4s} "
            f"{resname:>3s} "
            f"A{resseq:4d}    "
            f"{xA:8.3f}{yA:8.3f}{zA:8.3f}"
            f"  1.00  0.00          "
            f"{element:>2s}"
        )

        lines.append(line)
        serial += 1

    lines.append("END")
    pdb_text = "\n".join(lines) + "\n"

    print(
        f"[nma_bio3d] Generated PDB string with "
        f"{serial - 1} atoms from MDTraj trajectory"
    )

    return pdb_text


def check_r_environment() -> None:
    """
    Print and validate the R environment before Bio3D NMA.
    """
    r_code = r"""
    cat("[nma_bio3d] R version:", R.version.string, "\n")
    cat("[nma_bio3d] .libPaths():", paste(.libPaths(), collapse = " | "), "\n")

    rcpp_ok <- requireNamespace("Rcpp", quietly = TRUE)
    bio3d_ok <- requireNamespace("bio3d", quietly = TRUE)

    cat("[nma_bio3d] Rcpp available:", rcpp_ok, "\n")
    cat("[nma_bio3d] bio3d available:", bio3d_ok, "\n")

    if (!rcpp_ok) {
      stop("Rcpp is not available in this R environment")
    }

    if (!bio3d_ok) {
      stop("bio3d is not available in this R environment")
    }

    suppressPackageStartupMessages({
      library(Rcpp)
      library(bio3d)
    })

    cat("[nma_bio3d] Rcpp path:", system.file(package = "Rcpp"), "\n")
    cat("[nma_bio3d] bio3d path:", system.file(package = "bio3d"), "\n")
    """

    ro.r(r_code)


def setup_r_nma_function() -> None:
    """
    Define the R-side Bio3D aanma.pdb wrapper.

    Important:
    - No atom.select(pdb, "protein")
    - No trim.pdb()
    - No Bio3D-side atom filtering
    - outmodes = "all"

    This forces Bio3D to use exactly the atom set provided by Python.
    """
    r_code = r"""
    suppressPackageStartupMessages({
      library(bio3d)
    })

    run_aanma <- function(pdb_text) {

      tf <- tempfile(fileext = ".pdb")
      writeLines(pdb_text, tf)

      pdb <- tryCatch(
        read.pdb(tf),
        error = function(e) {
          unlink(tf)
          stop("read.pdb failed: ", e$message)
        }
      )

      unlink(tf)

      if (is.null(pdb)) {
        stop("read.pdb returned NULL")
      }

      if (is.null(pdb$atom)) {
        stop("pdb$atom is NULL")
      }

      if (nrow(pdb$atom) == 0) {
        stop("pdb contains 0 atoms")
      }

      cat("[nma_bio3d] Bio3D read.pdb atoms:", nrow(pdb$atom), "\n")

      # Use exactly the atoms provided by Python.
      # Do not call atom.select(pdb, 'protein'), because Bio3D may silently
      # remove caps, unusual residues, or atoms it does not classify as protein.
      pdb_trimmed <- pdb

      nma_aa <- tryCatch(
        aanma.pdb(
          pdb_trimmed,
          rtb = FALSE,
          outmodes = atom.select(pdb_trimmed, "all")
        ),
        error = function(e) {
          stop("aanma.pdb failed: ", e$message)
        }
      )

      if (is.null(nma_aa)) {
        stop("aanma.pdb returned NULL")
      }

      if (is.null(nma_aa$U)) {
        stop("nma_aa$U is NULL")
      }

      if (is.null(nma_aa$L)) {
        stop("nma_aa$L eigenvalues are NULL")
      }

      return(list(
        U = nma_aa$U,
        values = nma_aa$L
      ))
    }
    """

    ro.r(r_code)


def get_default_nma_save_dir(pdb_code: str, base_dir: str | Path = "results") -> Path:
    """
    Recommended structure-level NMA output directory.

    Example
    -------
    results/8bdp/nma/

    NMA modes are structure-based, so this path deliberately does not
    include a replica number.
    """
    return Path(base_dir) / pdb_code / "nma"


def run_aanma_r_from_traj(
    traj_protein_heavy: md.Trajectory,
    n_modes_keep: int,
    save_raw_modes_dir: str | Path | None = None,
    save_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Bio3D aanma.pdb NMA via R on an MDTraj trajectory.

    Parameters
    ----------
    traj_protein_heavy : md.Trajectory
        Protein-heavy trajectory/reference structure sent to Bio3D.

    n_modes_keep : int
        Number of non-trivial internal modes to keep after skipping the
        first 6 rigid-body modes.

    save_raw_modes_dir : str | Path | None
        If provided, save the raw unweighted Bio3D mode matrix to this
        directory. The saved raw matrix has shape (3N, n_modes), exactly
        like Bio3D nma$U / nma$modes: columns are mode vectors.

    save_prefix : str | None
        Optional prefix for saved .npy files. Leave as None for clean
        structure-level filenames without replica labels.

    Returns
    -------
    modes : np.ndarray
        Shape (n_modes_keep, 3N). Internal modes only, after skipping
        the first 6 rigid-body modes. This preserves the old return format
        used by the existing scripts.

    eigvals : np.ndarray
        Shape (n_modes_total,). Eigenvalues in Bio3D order, including
        the first 6 rigid-body modes.

    Notes
    -----
    The saved raw mode matrices are NOT eigenvalue-weighted and NOT
    normalised by Python. Use these saved files for direct PCA/NMA
    dot-product work, then normalise columns only when computing overlaps.
    """
    print_header("Running R aanma.pdb NMA in Bio3D...")

    if traj_protein_heavy.n_frames < 1:
        raise ValueError("[nma_bio3d] traj_protein_heavy has no frames")

    if traj_protein_heavy.n_atoms < 1:
        raise ValueError("[nma_bio3d] traj_protein_heavy has no atoms")

    print(
        f"[nma_bio3d] Python trajectory atoms sent to Bio3D: "
        f"{traj_protein_heavy.n_atoms}"
    )

    pdb_text = traj_to_pdb_string(traj_protein_heavy)

    check_r_environment()
    setup_r_nma_function()

    r_run_aanma = ro.globalenv["run_aanma"]
    res = r_run_aanma(pdb_text)

    with localconverter(default_converter + numpy2ri.converter):
        U_r = res.rx2("U")
        vals_r = res.rx2("values")

    U = np.array(U_r, dtype=float)
    eigvals = np.array(vals_r, dtype=float).reshape(-1)

    if U.ndim != 2:
        raise ValueError(
            f"[nma_bio3d] Expected U to be 2D, got shape {U.shape}"
        )

    n_dof, n_modes_r = U.shape

    print(
        f"[nma_bio3d] Mode matrix U shape: "
        f"(3N_atoms={n_dof}, n_modes={n_modes_r})"
    )
    print(f"[nma_bio3d] Eigenvalues shape: {eigvals.shape}")

    expected_dof = traj_protein_heavy.n_atoms * 3

    if n_dof != expected_dof:
        bio3d_atoms = n_dof // 3

        raise ValueError(
            f"[nma_bio3d] DOF mismatch:\n"
            f"  Bio3D atoms: {bio3d_atoms}\n"
            f"  Python atoms: {traj_protein_heavy.n_atoms}\n"
            f"  Bio3D DOF: {n_dof}\n"
            f"  Expected DOF: {expected_dof}\n"
            f"  Difference: "
            f"{traj_protein_heavy.n_atoms - bio3d_atoms} atoms\n"
            f"This means Bio3D still changed the atom set internally."
        )

    start = 6
    end = min(start + n_modes_keep, n_modes_r)

    if end <= start:
        raise ValueError(
            f"[nma_bio3d] Requested {n_modes_keep} modes, "
            f"but only {n_modes_r} modes are available "
            f"after removing 6 rigid-body modes."
        )

    # ------------------------------------------------------------------
    # Raw unweighted Bio3D mode matrices
    # ------------------------------------------------------------------
    # U is the direct Bio3D output matrix, equivalent to nma$U / nma$modes:
    #   shape = (3N, n_modes)
    #   columns = mode vectors
    # This is the matrix you want for PCA/NMA dot products. It is not
    # eigenvalue-weighted and it is not transposed.
    raw_modes_all = U
    raw_internal_modes_keep = U[:, start:end]
    internal_eigvals_keep = eigvals[start:end]

    if save_raw_modes_dir is not None:
        save_dir = Path(save_raw_modes_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # These are structure-level NMA files. They are not replica-specific
        # unless you deliberately call this function on a replica-specific
        # reference structure. By default, no replica prefix is used.
        if save_prefix is None or save_prefix == "":
            prefix = ""
        else:
            prefix = f"{save_prefix}_"

        raw_all_path = save_dir / f"{prefix}raw_modes_all.npy"
        raw_keep_path = save_dir / f"{prefix}raw_internal_modes_keep.npy"
        eig_all_path = save_dir / f"{prefix}eigenvalues_all.npy"
        eig_keep_path = save_dir / f"{prefix}internal_eigenvalues_keep.npy"

        np.save(raw_all_path, raw_modes_all)
        np.save(raw_keep_path, raw_internal_modes_keep)
        np.save(eig_all_path, eigvals)
        np.save(eig_keep_path, internal_eigvals_keep)

        print("[nma_bio3d] Saved structure-level raw unweighted Bio3D mode matrices:")
        print(f"  all modes:        {raw_all_path}  shape={raw_modes_all.shape}")
        print(f"  kept internal:    {raw_keep_path}  shape={raw_internal_modes_keep.shape}")
        print(f"  all eigenvalues:  {eig_all_path}  shape={eigvals.shape}")
        print(f"  kept eigenvalues: {eig_keep_path}  shape={internal_eigvals_keep.shape}")

    # Preserve the original returned selected-mode format for old code:
    #   shape = (n_modes_keep, 3N)
    # Existing fitting scripts likely expect this orientation.
    modes = raw_internal_modes_keep.T

    print(
        f"[nma_bio3d] Selected modes "
        f"{start + 1}..{end} "
        f"(total {end - start} modes for comparison)"
    )

    return modes, eigvals
