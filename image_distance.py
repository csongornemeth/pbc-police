#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

from io_utils import get_pdb_dir, print_header

"""
Run example:
python image_distance.py --pdb 3b9c --group System
"""

GMX_DEFAULT = "/work001/software/gromacs-bekker-2025/build/bin/gmx"


def run_cmd(cmd, input_text=None):
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Failed: {' '.join(cmd)}")
    return result


def find_cleaned_xtcs(tmp_dir: Path, pdb_code: str):
    return sorted(tmp_dir.glob(f"cleaned_{pdb_code}_*.xtc"))


def extract_replica_id(xtc_path: Path, pdb_code: str) -> str:
    # cleaned_1abc_0.xtc -> 0
    stem = xtc_path.stem
    prefix = f"cleaned_{pdb_code}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected cleaned XTC name: {xtc_path.name}")
    return stem[len(prefix):]


def parse_xvg(xvg_path: Path):
    times = []
    dists = []

    with xvg_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            times.append(float(parts[0]))
            dists.append(float(parts[1]))

    return np.array(times), np.array(dists)


def summarise_distances(dists: np.ndarray):
    if len(dists) == 0:
        return {
            "n_frames": 0,
            "min_dist_nm": np.nan,
            "max_dist_nm": np.nan,
            "mean_dist_nm": np.nan,
            "median_dist_nm": np.nan,
            "p01_dist_nm": np.nan,
            "p05_dist_nm": np.nan,
            "p95_dist_nm": np.nan,
            "p99_dist_nm": np.nan,
        }

    return {
        "n_frames": len(dists),
        "min_dist_nm": float(np.min(dists)),
        "max_dist_nm": float(np.max(dists)),
        "mean_dist_nm": float(np.mean(dists)),
        "median_dist_nm": float(np.median(dists)),
        "p01_dist_nm": float(np.percentile(dists, 1)),
        "p05_dist_nm": float(np.percentile(dists, 5)),
        "p95_dist_nm": float(np.percentile(dists, 95)),
        "p99_dist_nm": float(np.percentile(dists, 99)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate periodic image minimum distances from cleaned XTC files."
    )
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--gmx", default=GMX_DEFAULT)
    parser.add_argument(
        "--group",
        default="System",
        help="Group name to pass to gmx mindist. Usually System for dummy.tpr.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip replicas whose XVG already exists",
    )
    args = parser.parse_args()

    pdb_code = args.pdb.lower()
    pdb_dir = get_pdb_dir(pdb_code)

    out_root = Path(f"results/{pdb_code}")
    tmp_dir = out_root / "tmp2"
    dummy_tpr = tmp_dir / "dummy.tpr"

    if not tmp_dir.exists():
        raise FileNotFoundError(f"tmp2 dir not found: {tmp_dir}")
    if not dummy_tpr.exists():
        raise FileNotFoundError(f"dummy.tpr not found: {dummy_tpr}")

    cleaned_xtcs = find_cleaned_xtcs(tmp_dir, pdb_code)
    if not cleaned_xtcs:
        raise FileNotFoundError(f"No cleaned trajectories found in {tmp_dir}")

    dist_dir = out_root / "periodic_image_distance_0512"
    dist_dir.mkdir(parents=True, exist_ok=True)

    print_header("Periodic image distance calculation")
    print(f"PDB dir:    {pdb_dir}")
    print(f"TMP dir:    {tmp_dir}")
    print(f"dummy.tpr:  {dummy_tpr}")
    print(f"Output dir: {dist_dir}")
    print(f"GROMACS:    {args.gmx}")
    print(f"Group:      {args.group}")
    print(f"Trajs:      {len(cleaned_xtcs)}")

    summary_rows = []

    for xtc in cleaned_xtcs:
        replica_id = extract_replica_id(xtc, pdb_code)

        print_header(f"Replica {replica_id}")

        xvg_out = dist_dir / f"pi_dist_{pdb_code}_{replica_id}.xvg"
        log_out = dist_dir / f"pi_dist_{pdb_code}_{replica_id}.log"

        if args.skip_existing and xvg_out.exists():
            print(f"[SKIP] Already exists: {xvg_out.name}")
            _, dists = parse_xvg(xvg_out)
            stats = summarise_distances(dists)
            summary_rows.append({
                "pdb": pdb_code,
                "replica": replica_id,
                "xtc": xtc.name,
                "xvg": xvg_out.name,
                **stats,
                "status": "skipped_existing",
            })
            continue

        cmd = [
            args.gmx, "mindist",
            "-s", str(dummy_tpr),
            "-f", str(xtc),
            "-pi",
            "-od", str(xvg_out),
        ]

        result = run_cmd(cmd, input_text=f"{args.group}\n{args.group}\n")

        with log_out.open("w") as fh:
            fh.write("COMMAND:\n")
            fh.write(" ".join(cmd) + "\n\n")
            fh.write("STDOUT:\n")
            fh.write(result.stdout or "")
            fh.write("\nSTDERR:\n")
            fh.write(result.stderr or "")

        _, dists = parse_xvg(xvg_out)
        stats = summarise_distances(dists)

        print(
            f"[OK] min={stats['min_dist_nm']:.4f} nm | "
            f"p05={stats['p05_dist_nm']:.4f} nm | "
            f"mean={stats['mean_dist_nm']:.4f} nm"
        )

        summary_rows.append({
            "pdb": pdb_code,
            "replica": replica_id,
            "xtc": xtc.name,
            "xvg": xvg_out.name,
            **stats,
            "status": "ok",
        })

    summary_csv = dist_dir / f"pi_dist_summary_{pdb_code}.csv"

    fieldnames = [
        "pdb",
        "replica",
        "xtc",
        "xvg",
        "n_frames",
        "min_dist_nm",
        "max_dist_nm",
        "mean_dist_nm",
        "median_dist_nm",
        "p01_dist_nm",
        "p05_dist_nm",
        "p95_dist_nm",
        "p99_dist_nm",
        "status",
    ]

    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print_header("DONE")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()