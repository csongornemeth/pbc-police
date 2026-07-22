#!/usr/bin/env python3

import subprocess
from pathlib import Path

GMX = "/work001/software/gromacs-bekker-2025/build/bin/gmx"

PROJECT_ROOT = Path("/work001/csongor/boxpred/pbc-police")
PDB_LIST = PROJECT_ROOT / "passing_pdbs.txt"

DB_ROOT = Path("/work001/misc/bekker/kakC/dynamicsdb/raw")
PHASE = "5"
REPLICA = "0"

GROUP = "Protein-H"

with PDB_LIST.open() as handle:
    pdbs = [
        line.strip().lower()
        for line in handle
        if line.strip() and not line.lstrip().startswith("#")
    ]

for pdb in pdbs:
    pdb_dir = DB_ROOT / PHASE / pdb
    build_dir = pdb_dir / "build"
    replica_dir = pdb_dir / "validation" / REPLICA

    gro = replica_dir / f"prod.part0001.gro"
    index = build_dir / "index.ndx"
    tpr = replica_dir / "prod.tpr"

    output = PROJECT_ROOT / "results" / pdb / f"target_{pdb}.pdb"
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"{pdb}: target already exists, skipping")
        continue

    missing = [path for path in (gro, index, tpr) if not path.is_file()]

    if missing:
        print(f"{pdb}: missing input:")
        for path in missing:
            print(f"  {path}")
        continue

    print(f"{pdb}: creating {output}")

    subprocess.run(
        [
            GMX,
            "trjconv",
            "-s",
            str(tpr),
            "-f",
            str(gro),
            "-n",
            str(index),
            "-o",
            str(output),
        ],
        input=f"{GROUP}\n",
        text=True,
        check=True,
    )

print("Done.")