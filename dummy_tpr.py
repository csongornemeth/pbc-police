#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from io_utils import get_pdb_dir, collect_xtc_paths, print_header

"""
Run examples:
python dummy_tpr.py --pdb 1jka
python dummy_tpr.py --pdb 7hzc --group Protein-H
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
        print("\n[CMD FAILED]")
        print("Command:", " ".join(cmd))
        print("\n[STDOUT]")
        print(result.stdout)
        print("\n[STDERR]")
        print(result.stderr)
        raise RuntimeError(f"Failed: {' '.join(cmd)}")
    return result


def run_cmd_result(cmd, input_text=None):
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )


# -------------------------
# CLEANUP LOGIC
# -------------------------
def cleanup_tmp_dir(tmp_dir: Path, pdb_code: str, wipe: bool):
    print_header("Cleanup")

    if not tmp_dir.exists():
        return

    if wipe:
        print("[INFO] Wiping entire tmp directory")
        for f in tmp_dir.glob("*"):
            try:
                f.unlink()
            except IsADirectoryError:
                pass
        return

    removed = 0

    for f in tmp_dir.glob("*"):
        name = f.name

        if name.startswith(f"{pdb_code}_") and name.endswith(".xtc"):
            print(f"[CLEAN] Removing temp: {name}")
            f.unlink()
            removed += 1

        elif name == "dummy.tpr":
            print("[CLEAN] Removing old dummy.tpr")
            f.unlink()
            removed += 1

    print(f"[INFO] Removed {removed} leftover files")


# -------------------------
def make_dummy_tpr(gmx, build_dir, dummy_tpr, group_name):
    print_header(f"Creating dummy.tpr with group {group_name}")

    cmd = [
        gmx, "convert-tpr",
        "-s", str(build_dir / "nvt.tpr"),
        "-n", str(build_dir / "index.ndx"),
        "-o", str(dummy_tpr),
    ]
    run_cmd(cmd, input_text=f"{group_name}\n")


def group_xtcs_by_replica(xtc_paths):
    replica_groups = defaultdict(list)

    for xtc in xtc_paths:
        replica_id = xtc.parent.name
        replica_groups[replica_id].append(xtc)

    for r in replica_groups:
        replica_groups[r] = sorted(replica_groups[r])

    return replica_groups


def parse_index_group_sizes(index_file: Path) -> dict[str, int]:
    groups = {}
    current_group = None
    current_count = 0

    with open(index_file) as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                if current_group is not None:
                    groups[current_group] = current_count

                current_group = line.strip("[] ").strip()
                current_count = 0
            else:
                if current_group is not None:
                    current_count += len(line.split())

    if current_group is not None:
        groups[current_group] = current_count

    return groups


def get_xtc_atom_count(gmx: str, xtc_path: Path) -> int:
    result = run_cmd_result([gmx, "check", "-f", str(xtc_path)])

    text = result.stdout + "\n" + result.stderr

    patterns = [
        r"#\s*Atoms\s+(\d+)",
        r"natoms\s*=\s*(\d+)",
        r"contains\s+(\d+)\s+atoms",
        r"(\d+)\s+atoms",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    print("[DEBUG] gmx check output:")
    print(text)

    raise RuntimeError(
        f"Could not determine atom count from XTC: {xtc_path}"
    )


def detect_matching_group(gmx: str, build_dir: Path, replica_groups, user_group: str | None):
    print_header("Detecting correct group")

    if not replica_groups:
        raise RuntimeError("No XTC files found")

    first_replica = sorted(replica_groups, key=lambda x: int(x))[0]
    first_xtc = replica_groups[first_replica][0]

    xtc_natoms = get_xtc_atom_count(gmx, first_xtc)
    print(f"[INFO] XTC atom count: {xtc_natoms}")

    index_file = build_dir / "index.ndx"
    group_sizes = parse_index_group_sizes(index_file)

    candidate_groups = ["Protein-H", "solute-H", "System"]
    if user_group is not None:
        candidate_groups = [user_group]

    print("[INFO] Candidate group sizes:")
    for g in candidate_groups:
        if g in group_sizes:
            print(f"  {g}: {group_sizes[g]}")
        else:
            print(f"  {g}: not present")

    matches = [g for g in candidate_groups if group_sizes.get(g) == xtc_natoms]

    if len(matches) == 1:
        print(f"[AUTO] Using group: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        print(f"[WARN] Multiple matching groups found: {', '.join(matches)}")
        print(f"[AUTO] Using first match: {matches[0]}")
        return matches[0]

    raise RuntimeError(
        f"No matching group found for XTC atom count {xtc_natoms}. "
        f"Candidate sizes: " +
        ", ".join(f"{g}={group_sizes.get(g, 'missing')}" for g in candidate_groups)
    )


def clean_replica(gmx, pdb_code, replica_id, xtc_files, dummy_tpr, group_name, tmp_dir):
    print_header(f"Replica {replica_id}")

    temp_files = []

    for i, xtc in enumerate(xtc_files):
        mol_xtc = tmp_dir / f"{pdb_code}_{replica_id}_{i}_mol.xtc"
        tmp_xtc = tmp_dir / f"{pdb_code}_{replica_id}_{i}.xtc"

        print(f"[INFO] {xtc.name} -> {tmp_xtc.name}")

        # Step 1: make molecules whole
        run_cmd(
            [
                gmx, "trjconv",
                "-s", str(dummy_tpr),
                "-f", str(xtc),
                "-o", str(mol_xtc),
                "-pbc", "whole",
            ],
            input_text=f"System\nSystem\n",
        )

        # Step 2: cluster chains together
        run_cmd(
            [
                gmx, "trjconv",
                "-s", str(dummy_tpr),
                "-f", str(mol_xtc),
                "-o", str(tmp_xtc),
                "-pbc", "nojump",
                "-center",
            ],
            input_text=f"System\nSystem\n",
        )

        temp_files.append(str(tmp_xtc))

        # cleanup intermediate
        try:
            os.remove(mol_xtc)
        except FileNotFoundError:
            pass

    out_xtc = tmp_dir / f"cleaned_{pdb_code}_{replica_id}.xtc"

    run_cmd([gmx, "trjcat", "-f", *temp_files, "-o", str(out_xtc)])

    for f in temp_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print(f"[OK] {out_xtc}")
    return out_xtc


# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--group", choices=["Protein-H", "solute-H", "System"], default=None)
    parser.add_argument("--gmx", default=GMX_DEFAULT)
    parser.add_argument("--wipe", action="store_true", help="Delete all files in tmp before running")

    args = parser.parse_args()

    pdb_code = args.pdb.lower()
    pdb_dir = get_pdb_dir(pdb_code)
    build_dir = pdb_dir / "build"

    out_root = Path(f"results/{pdb_code}")
    tmp_dir = out_root / "tmp2"

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cleanup_tmp_dir(tmp_dir, pdb_code, wipe=args.wipe)

    print_header("Paths")
    print(f"PDB dir: {pdb_dir}")
    print(f"TMP dir: {tmp_dir}")
    print(f"GROMACS: {args.gmx}")

    xtc_paths = collect_xtc_paths(pdb_dir)
    replica_groups = group_xtcs_by_replica(xtc_paths)

    group_name = detect_matching_group(
        args.gmx,
        build_dir,
        replica_groups,
        args.group,
    )

    print(f"Group:   {group_name}")

    dummy_tpr = tmp_dir / "dummy.tpr"
    make_dummy_tpr(args.gmx, build_dir, dummy_tpr, group_name)

    for replica_id in sorted(replica_groups, key=lambda x: int(x)):
        clean_replica(
            args.gmx,
            pdb_code,
            replica_id,
            replica_groups[replica_id],
            dummy_tpr,
            group_name,
            tmp_dir,
        )

    print_header("DONE")


if __name__ == "__main__":
    main()