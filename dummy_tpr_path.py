#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

"""
Clean/repair replica XTCs for the new solvated systems.

Expected layout:

solvate_trial/{pdb}/
├── replicas/
│   ├── rep1/
│   │   ├── prod_rep1.xtc
│   │   ├── prod_rep1.tpr
│   ├── rep2/
│   │   ├── prod_rep2.xtc
│   │   ├── prod_rep2.tpr
│   └── ...
├── replica_index.ndx
├── rebuilt_autolig_build/
└── tmp/

Run examples:

python clean_solvate_trial_xtcs.py --pdb 7lak --group solute-H --wipe

python dummy_tpr_path.py \
  --pdb 7lak \
  --xtc-pattern "validation/*/prod.xtc" \
  --group solute-H \
  --wipe

python clean_solvate_trial_xtcs.py \
  --pdb 7lak \
  --xtc-pattern "validation/*/prod.xtc" \
  --group solute-H \
  --wipe

python dummy_tpr_path.py \
  --pdb 7n6h \
  --root MD_sims/7n6h_10ns_newbox \
  --group solute-H \
  --xtc-pattern "prod.xtc" \
  --wipe
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


def run_cmd_result(cmd, input_text=None):
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )


def rep_sort_key(replica_id: str):
    """
    Sorts:
      rep1, rep2, ..., rep10
    correctly.
    """
    m = re.search(r"(\d+)$", replica_id)
    if m:
        return int(m.group(1))
    return replica_id


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

    raise RuntimeError(f"Could not determine atom count from XTC: {xtc_path}")


def group_xtcs_by_replica(xtc_paths):
    replica_groups = defaultdict(list)

    for xtc in xtc_paths:
        replica_id = xtc.parent.name
        replica_groups[replica_id].append(xtc)

    for r in replica_groups:
        replica_groups[r] = sorted(replica_groups[r])

    return replica_groups


def find_source_tpr(out_root: Path, replica_groups) -> Path:
    """
    Use first available replica TPR.
    Priority:
      prod.tpr
      md.tpr
      prep.tpr
      *.tpr
    """
    first_replica = sorted(replica_groups, key=rep_sort_key)[0]
    rep_dir = out_root / "validation" / first_replica

    patterns = [
        "prod.tpr",
        "md.tpr",
        "prep.tpr",
        "*.tpr",
    ]

    for pattern in patterns:
        hits = sorted(rep_dir.glob(pattern))
        if hits:
            return hits[0]

    raise RuntimeError(f"No .tpr file found in {rep_dir}")


def detect_matching_group(
    gmx: str,
    index_file: Path,
    replica_groups,
    user_group: str | None,
):
    print_header("Detecting correct group")

    if not replica_groups:
        raise RuntimeError("No XTC files found")

    first_replica = sorted(replica_groups, key=rep_sort_key)[0]
    first_xtc = replica_groups[first_replica][0]

    xtc_natoms = get_xtc_atom_count(gmx, first_xtc)

    print(f"[INFO] First replica: {first_replica}")
    print(f"[INFO] First XTC: {first_xtc}")
    print(f"[INFO] XTC atom count: {xtc_natoms}")

    group_sizes = parse_index_group_sizes(index_file)

    candidate_groups = ["solute-H", "Protein-H", "solute", "System"]
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
        f"Candidate sizes: "
        + ", ".join(f"{g}={group_sizes.get(g, 'missing')}" for g in candidate_groups)
    )


def make_dummy_tpr(
    gmx: str,
    source_tpr: Path,
    index_file: Path,
    dummy_tpr: Path,
    group_name: str,
):
    print_header(f"Creating dummy.tpr with group {group_name}")

    cmd = [
        gmx,
        "convert-tpr",
        "-s",
        str(source_tpr),
        "-n",
        str(index_file),
        "-o",
        str(dummy_tpr),
    ]

    run_cmd(cmd, input_text=f"{group_name}\n")


def clean_replica(
    gmx: str,
    pdb_code: str,
    replica_id: str,
    xtc_files,
    dummy_tpr: Path,
    tmp_dir: Path,
):
    print_header(f"Replica {replica_id}")

    temp_files = []

    for i, xtc in enumerate(xtc_files):
        mol_xtc = tmp_dir / f"{pdb_code}_{replica_id}_{i}_mol.xtc"
        tmp_xtc = tmp_dir / f"{pdb_code}_{replica_id}_{i}.xtc"

        print(f"[INFO] {xtc.name} -> {tmp_xtc.name}")

        # Step 1: make molecules whole
        run_cmd(
            [
                gmx,
                "trjconv",
                "-s",
                str(dummy_tpr),
                "-f",
                str(xtc),
                "-o",
                str(mol_xtc),
                "-pbc",
                "whole",
            ],
            input_text="System\nSystem\n",
        )

        run_cmd(
            [
                gmx,
                "trjconv",
                "-s",
                str(dummy_tpr),
                "-f",
                str(xtc),
                "-o",
                str(mol_xtc),
                "-pbc",
                "whole",
            ],
            input_text="System\nSystem\n",
        )

        # Step 2: remove jumps and center
        run_cmd(
            [
                gmx,
                "trjconv",
                "-s",
                str(dummy_tpr),
                "-f",
                str(mol_xtc),
                "-o",
                str(tmp_xtc),
                "-pbc",
                "nojump",
                "-center",
            ],
            input_text="System\nSystem\n",
        )

        temp_files.append(str(tmp_xtc))

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", required=True)

    parser.add_argument(
        "--root",
        default=f"MD_sims/7n6h_10ns_newbox",
        help="Root folder containing {pdb}/replicas. Default: solvate_trial",
    )

    parser.add_argument(
        "--xtc-pattern",
        default="prod_rep*.xtc",
        help="XTC pattern inside each replica folder. Default: prod_rep*.xtc",
    )

    parser.add_argument(
        "--group",
        choices=["Protein-H", "solute-H", "solute", "System"],
        default=None,
        help="Force group. If omitted, auto-detect from XTC atom count.",
    )

    parser.add_argument("--gmx", default=GMX_DEFAULT)

    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all files in tmp before running",
    )

    args = parser.parse_args()

    pdb_code = args.pdb.lower()

    out_root = Path(args.root)
    replicas_dir = out_root / "validation"
    index_file = out_root / "build" / "index.ndx"
    tmp_dir = out_root / "cleaned_xtcs"

    if not out_root.exists():
        raise RuntimeError(f"System folder not found: {out_root}")

    if not replicas_dir.exists():
        raise RuntimeError(f"Replicas directory not found: {replicas_dir}")

    if not index_file.exists():
        raise RuntimeError(f"Index file not found: {index_file}")

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cleanup_tmp_dir(tmp_dir, pdb_code, wipe=args.wipe)

    print_header("Paths")
    print(f"PDB/system: {pdb_code}")
    print(f"Root:       {out_root}")
    print(f"Replicas:   {replicas_dir}")
    print(f"Index:      {index_file}")
    print(f"TMP dir:    {tmp_dir}")
    print(f"GROMACS:    {args.gmx}")
    print(f"Pattern:    {args.xtc_pattern}")

    xtc_paths = sorted(replicas_dir.glob(f"*/{args.xtc_pattern}"))

    if not xtc_paths:
        raise RuntimeError(
            f"No XTC files found with pattern: {replicas_dir}/*/{args.xtc_pattern}"
        )

    print_header("Found XTC files")
    for x in xtc_paths:
        print(x)

    replica_groups = group_xtcs_by_replica(xtc_paths)

    group_name = detect_matching_group(
        args.gmx,
        index_file,
        replica_groups,
        args.group,
    )

    print(f"Group: {group_name}")

    source_tpr = find_source_tpr(out_root, replica_groups)
    print(f"Source TPR: {source_tpr}")

    dummy_tpr = tmp_dir / "dummy.tpr"

    make_dummy_tpr(
        args.gmx,
        source_tpr,
        index_file,
        dummy_tpr,
        group_name,
    )

    for replica_id in sorted(replica_groups, key=rep_sort_key):
        clean_replica(
            args.gmx,
            pdb_code,
            replica_id,
            replica_groups[replica_id],
            dummy_tpr,
            tmp_dir,
        )

    print_header("DONE")


if __name__ == "__main__":
    main()