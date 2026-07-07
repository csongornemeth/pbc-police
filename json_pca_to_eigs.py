from __future__ import annotations

import json
import gzip
import re
from pathlib import Path
import numpy as np


IPCA_RE = re.compile(r"(?P<pdb>[0-9a-zA-Z]{4})_rep(?P<rep>\d+)_ipca\.json\.gz$")


def load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def extract_pca_data(pca: dict):
    if "components_" not in pca:
        raise KeyError("Missing key: components_")

    V = np.asarray(pca["components_"], dtype=float)

    if "explained_variance_" in pca and pca["explained_variance_"] is not None:
        lambdas = np.asarray(pca["explained_variance_"], dtype=float)

    elif "singular_values_" in pca and pca["singular_values_"] is not None:
        if "n_samples_" not in pca:
            raise KeyError("singular_values_ present but n_samples_ missing")
        s = np.asarray(pca["singular_values_"], dtype=float)
        n = int(pca["n_samples_"])
        lambdas = (s ** 2) / (n - 1)

    else:
        raise KeyError("No explained_variance_ or singular_values_ found")

    return V, lambdas


def process_ipca_file(ipca_path: Path):
    m = IPCA_RE.match(ipca_path.name)
    if not m:
        return  # silently skip non-matching files

    pdb = m.group("pdb")
    rep = m.group("rep")

    outdir = ipca_path.parent / f"rep{rep}_pca_eigs"

    pca = load_json_gz(ipca_path)
    V, lambdas = extract_pca_data(pca)

    outdir.mkdir(exist_ok=True)
    np.save(outdir / "eigenvectors.npy", V)
    np.save(outdir / "eigenvalues.npy", lambdas)

    print(f"{pdb} rep{rep}: V{V.shape}, λ{lambdas.shape}")


def batch_build_pca_eigensystems(results_root: Path):
    for ipca_file in results_root.rglob("*_rep*_ipca.json.gz"):
        process_ipca_file(ipca_file)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results/ directory")
    args = ap.parse_args()

    batch_build_pca_eigensystems(Path(args.results))
