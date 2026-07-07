#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

"""
Calculate periodic image minimum distances from cleaned XTC files.

Expected layout:

MD_sims/{system_name}/
├── cleaned_xtcs/
│   ├── dummy.tpr
│   ├── cleaned_7n6h_rep1.xtc
│   ├── cleaned_7n6h_rep2.xtc
│   └── ...
└── periodic_image_distance/

Run example:

python image_distance_path.py \
  --pdb 7n6h_10ns_newbox \
  --group System

or explicitly:

python image_distance_path.py \
  --root MD_sims/7n6h_10ns_newbox \
  --group System
"""

GMX_DEFAULT = "/work001/software/gromacs-bekker-2025/build/bin/gmx"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_cmd(cmd, input_text=None):
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("\n[CMD FAILED]")
        print("Command:", " ".join(map(str, cmd)))

        print("\n[STDOUT]")
        print(result.stdout)

        print("\n[STDERR]")
        print(result.stderr)

        raise RuntimeError(f"Failed: {' '.join(map(str, cmd))}")

    return result


def find_cleaned_xtcs(tmp_dir: Path):
    """
    Find cleaned trajectories in cleaned_xtcs/.

    Accepts:
      cleaned_7n6h_rep1.xtc
      cleaned_7n6h_rep2.xtc
      cleaned_7n6h_0.xtc
      etc.

    Excludes intermediate temp files if any are left.
    """
    xtcs = sorted(tmp_dir.glob("cleaned_*.xtc"))

    # Safety filter: avoid accidentally reading non-final temp files
    xtcs = [
        x for x in xtcs
        if not x.name.endswith("_mol.xtc")
    ]

    return xtcs


def extract_replica_id(xtc_path: Path) -> str:
    """
    Extract replica ID from cleaned XTC filename.

    Examples:
      cleaned_7n6h_rep1.xtc -> rep1
      cleaned_7n6h_rep10.xtc -> rep10
      cleaned_7n6h_0.xtc -> 0
      cleaned_7n6h_10ns_newbox_rep1.xtc -> rep1
    """
    stem = xtc_path.stem

    if not stem.startswith("cleaned_"):
        raise ValueError(f"Unexpected cleaned XTC name: {xtc_path.name}")

    rest = stem[len("cleaned_"):]

    # Usually the replica is the final underscore-separated part
    # cleaned_7n6h_rep1 -> rep1
    # cleaned_7n6h_10ns_newbox_rep1 -> rep1
    parts = rest.split("_")

    if len(parts) < 2:
        raise ValueError(f"Could not extract replica id from: {xtc_path.name}")

    return parts[-1]


def parse_xvg(xvg_path: Path):
    times = []
    dists = []

    with xvg_path.open() as fh:
        for line in fh:
            line = line.strip()

            if not line:
                continue

            if line.startswith("@") or line.startswith("#"):
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
        "n_frames": int(len(dists)),
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

    parser.add_argument(
        "--pdb",
        default=None,
        help=(
            "System folder name under MD_sims/, e.g. 7n6h_10ns_newbox. "
            "Not required if --root is given."
        ),
    )

    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Full system root folder, e.g. MD_sims/7n6h_10ns_newbox. "
            "If given, this overrides --pdb."
        ),
    )

    parser.add_argument(
        "--gmx",
        default=GMX_DEFAULT,
        help="Path to GROMACS executable.",
    )

    parser.add_argument(
        "--group",
        default="System",
        help="Group name to pass to gmx mindist. Usually System for dummy.tpr.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip replicas whose XVG already exists.",
    )

    args = parser.parse_args()

    if args.root is not None:
        out_root = Path(args.root)
        system_name = out_root.name.lower()
    else:
        if args.pdb is None:
            raise RuntimeError("You must give either --pdb or --root.")
        system_name = args.pdb.lower()
        out_root = Path("MD_sims") / system_name

    tmp_dir = out_root / "cleaned_xtcs"
    dummy_tpr = tmp_dir / "dummy.tpr"

    if not out_root.exists():
        raise FileNotFoundError(f"System root not found: {out_root}")

    if not tmp_dir.exists():
        raise FileNotFoundError(f"cleaned_xtcs dir not found: {tmp_dir}")

    if not dummy_tpr.exists():
        raise FileNotFoundError(f"dummy.tpr not found: {dummy_tpr}")

    cleaned_xtcs = find_cleaned_xtcs(tmp_dir)

    if not cleaned_xtcs:
        raise FileNotFoundError(f"No cleaned trajectories found in {tmp_dir}")

    dist_dir = out_root / "periodic_image_distance"
    dist_dir.mkdir(parents=True, exist_ok=True)

    print_header("Periodic image distance calculation")
    print(f"System:     {system_name}")
    print(f"Root:       {out_root}")
    print(f"TMP dir:    {tmp_dir}")
    print(f"dummy.tpr:  {dummy_tpr}")
    print(f"Output dir: {dist_dir}")
    print(f"GROMACS:    {args.gmx}")
    print(f"Group:      {args.group}")
    print(f"Trajs:      {len(cleaned_xtcs)}")

    print_header("Found cleaned XTCs")
    for x in cleaned_xtcs:
        print(x)

    summary_rows = []

    for xtc in cleaned_xtcs:
        replica_id = extract_replica_id(xtc)

        print_header(f"Replica {replica_id}")

        xvg_out = dist_dir / f"pi_dist_{system_name}_{replica_id}.xvg"
        log_out = dist_dir / f"pi_dist_{system_name}_{replica_id}.log"

        if args.skip_existing and xvg_out.exists():
            print(f"[SKIP] Already exists: {xvg_out.name}")

            _, dists = parse_xvg(xvg_out)
            stats = summarise_distances(dists)

            summary_rows.append({
                "system": system_name,
                "replica": replica_id,
                "xtc": xtc.name,
                "xvg": xvg_out.name,
                **stats,
                "status": "skipped_existing",
            })

            continue

        cmd = [
            args.gmx,
            "mindist",
            "-s",
            str(dummy_tpr),
            "-f",
            str(xtc),
            "-pi",
            "-od",
            str(xvg_out),
        ]

        result = run_cmd(
            cmd,
            input_text=f"{args.group}\n{args.group}\n",
        )

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
            "system": system_name,
            "replica": replica_id,
            "xtc": xtc.name,
            "xvg": xvg_out.name,
            **stats,
            "status": "ok",
        })

    summary_csv = dist_dir / f"pi_dist_summary_{system_name}.csv"

    fieldnames = [
        "system",
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