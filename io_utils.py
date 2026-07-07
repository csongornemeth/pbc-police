# io_utils.py  (cluster DB layout)
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import mdtraj as md

#Printing helper
def print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


#DB path configuration

RAW_ROOT = Path("/work001/misc/bekker/kakC/dynamicsdb/raw")
PHASES = {"1", "2", "2b", "3", "3b", "4", "4b", "4c", "4d", "4f", "5"}

_part_re = re.compile(r"prod\.part(\d+)\.xtc$")


def _require_exists(p: Path, what: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")
    return p


#Order the phases so they can be used later to chose which pdb is newer

def phase_order(ph: str) -> tuple[int, int]:
    """
    Helper to sort phases naturally (newer = larger):

      1 < 2 < 2b < 3 < 3b < 4 < 4b < 4c < 4d < 4f < 5
    """
    ph = str(ph).strip().lower()
    m = re.fullmatch(r"(\d+)([a-z]?)", ph)
    if not m:
        return (-1, -1)

    num = int(m.group(1))
    suf = m.group(2) or ""

    # suffix ranking: "" < b < c < d < f
    suf_rank = {"": 0, "b": 1, "c": 2, "d": 3, "f": 4}.get(suf, 99)
    return (num, suf_rank)


def _pick_newest_by_phase(candidates: list[Path]) -> Path:
    """
    candidates are RAW_ROOT/<phase>/<pdb_code>.
    Pick the newest by phase_order().
    """

    def phase_of(p: Path) -> str:
        # RAW_ROOT/<phase>/<pdb_code>
        return p.parent.name

    ranked = sorted(candidates, key=lambda p: phase_order(phase_of(p)), reverse=True)

    if len(ranked) > 1:
        print_header("[WARN] Duplicate PDB found in multiple phases; picking newest phase")
        for d in ranked:
            ph = d.parent.name
            print(f"  - phase={ph:>2}  key={phase_order(ph)}  dir={d}")
        print(f"[WARN] Using: {ranked[0]}")

    return ranked[0]


#get the pdb directory path

# Compute once: newest -> oldest
PHASES_NEWEST_FIRST = tuple(sorted(PHASES, key=phase_order, reverse=True))

@lru_cache(maxsize=None)
def get_pdb_dir(pdb_code: str, phase: str | None = None) -> Path:
    """
    Return: RAW_ROOT/<phase>/<pdb_code>

    If phase is None, find the newest phase that contains the pdb_code
    by scanning phases from newest -> oldest and returning the first hit.
    """
    pdb_code = pdb_code.strip().lower()

    if phase is not None:
        phase = str(phase).strip()
        if phase not in PHASES:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {sorted(PHASES)}")
        return _require_exists(RAW_ROOT / phase / pdb_code, "PDB directory")

    # Fast path: newest -> oldest, first hit wins (minimal filesystem stats)
    for ph in PHASES_NEWEST_FIRST:
        p = RAW_ROOT / ph / pdb_code
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find pdb_code='{pdb_code}' under {RAW_ROOT} in any phase."
    )


    return _pick_newest_by_phase(candidates)

def get_topology_path(pdb_dir: Path) -> Path:
    """
    Topology is in:
      pdb_dir/build/npt.gro
    """
    gro = pdb_dir / "build" / "npt.gro"
    return _require_exists(gro, "topology (npt.gro)")


#get the trajectories

def collect_xtc_paths(pdb_dir: Path) -> list[Path]:
    """
    Collect *all* XTC parts from replicas 0..9:
      pdb_dir/validation/{replica_id}/prod.part*.xtc

    Sorted:
      replica 0 parts -> replica 1 parts -> ... -> replica 9 parts
      and within replica numeric by prod.part index.
    """
    xtc_paths: list[Path] = []
    val_root = pdb_dir / "validation"
    _require_exists(val_root, "validation directory")

    for i in range(10):
        traj_dir = val_root / str(i)
        if not traj_dir.exists():
            print(f"[WARN] Missing replica directory: {traj_dir}")
            continue

        parts = list(traj_dir.glob("prod.part*.xtc"))
        if not parts:
            print(f"[WARN] No trajectory files found in: {traj_dir}")
            continue

        def key(p: Path) -> int:
            m = _part_re.search(p.name)
            return int(m.group(1)) if m else 10**9

        xtc_paths.extend(sorted(parts, key=key))

    if not xtc_paths:
        raise RuntimeError(f"No XTC files found under {val_root}/0..9. Cannot run PCA.")

    print_header("Trajectory files discovered (all replicas)")
    for x in xtc_paths:
        print(f"    {x}")
    return xtc_paths


#load full topology
def load_full_topology(pdb_dir: Path) -> tuple[md.Trajectory, md.Topology]:
    """
    Load full-system topology that matches the XTC atom ordering.
    Using npt.gro (build) as per DB spec.
    """
    gro = get_topology_path(pdb_dir)
    traj_full = md.load(gro.as_posix())
    top_full = traj_full.topology
    print(f"[MAIN] Full topology natoms ({gro.name}): {top_full.n_atoms}")
    return traj_full, top_full


def get_protein_heavy_indices(top_full: md.Topology):
    """Return indices of protein heavy atoms on the full topology."""
    protein_heavy_idx = top_full.select("protein and not element H")
    print(f"[MAIN] Protein heavy-atom count: {protein_heavy_idx.size}")
    return protein_heavy_idx
