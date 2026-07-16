#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from io_utils import collect_xtc_paths, get_pdb_dir, print_header


"""
Example:

python image_distance.py --pdb 7hzc --threshold 1.2

Optionally force the matching group:

python pidist.py --pdb 7hzc --threshold 1.2

"""


GMX_DEFAULT = "/work001/software/gromacs-bekker-2025/build/bin/gmx"


# ---------------------------------------------------------------------
# GENERAL COMMAND HELPERS
# ---------------------------------------------------------------------

def run_cmd(
    cmd: list[str],
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
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


def run_cmd_result(
    cmd: list[str],
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------

def cleanup_tmp_dir(
    tmp_dir: Path,
    pdb_code: str,
    wipe: bool,
) -> None:
    print_header("Cleanup")

    if not tmp_dir.exists():
        return

    if wipe:
        print("[INFO] Wiping entire temporary directory")

        for path in tmp_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()

        return

    removed = 0

    removable_patterns = [
        f"{pdb_code}_*.xtc",
        f"cleaned_{pdb_code}_*.xtc",
        "dummy.tpr",
    ]

    for pattern in removable_patterns:
        for path in tmp_dir.glob(pattern):
            if not path.is_file():
                continue

            print(f"[CLEAN] Removing: {path.name}")
            path.unlink()
            removed += 1

    print(f"[INFO] Removed {removed} leftover files")


# ---------------------------------------------------------------------
# DUMMY TPR AND TRAJECTORY PREPARATION
# ---------------------------------------------------------------------

def make_dummy_tpr(
    gmx: str,
    build_dir: Path,
    dummy_tpr: Path,
    group_name: str,
) -> None:
    print_header(f"Creating dummy.tpr with group {group_name}")

    nvt_tpr = build_dir / "nvt.tpr"
    index_file = build_dir / "index.ndx"

    if not nvt_tpr.exists():
        raise FileNotFoundError(f"nvt.tpr not found: {nvt_tpr}")

    if not index_file.exists():
        raise FileNotFoundError(f"index.ndx not found: {index_file}")

    cmd = [
        gmx,
        "convert-tpr",
        "-s",
        str(nvt_tpr),
        "-n",
        str(index_file),
        "-o",
        str(dummy_tpr),
    ]

    run_cmd(cmd, input_text=f"{group_name}\n")


def group_xtcs_by_replica(
    xtc_paths: list[Path],
) -> dict[str, list[Path]]:
    replica_groups: dict[str, list[Path]] = defaultdict(list)

    for xtc in xtc_paths:
        replica_id = xtc.parent.name
        replica_groups[replica_id].append(xtc)

    for replica_id in replica_groups:
        replica_groups[replica_id] = sorted(
            replica_groups[replica_id]
        )

    return dict(replica_groups)


def parse_index_group_sizes(
    index_file: Path,
) -> dict[str, int]:
    groups: dict[str, int] = {}

    current_group: str | None = None
    current_count = 0

    with index_file.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                if current_group is not None:
                    groups[current_group] = current_count

                current_group = line.strip("[] ").strip()
                current_count = 0

            elif current_group is not None:
                current_count += len(line.split())

    if current_group is not None:
        groups[current_group] = current_count

    return groups


def get_xtc_atom_count(
    gmx: str,
    xtc_path: Path,
) -> int:
    result = run_cmd_result(
        [
            gmx,
            "check",
            "-f",
            str(xtc_path),
        ]
    )

    text = (result.stdout or "") + "\n" + (result.stderr or "")

    patterns = [
        r"#\s*Atoms\s+(\d+)",
        r"natoms\s*=\s*(\d+)",
        r"contains\s+(\d+)\s+atoms",
        r"(\d+)\s+atoms",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return int(match.group(1))

    print("[DEBUG] gmx check output:")
    print(text)

    raise RuntimeError(
        f"Could not determine atom count from XTC: {xtc_path}"
    )


def replica_sort_key(replica_id: str):
    try:
        return 0, int(replica_id)
    except ValueError:
        return 1, replica_id


def detect_matching_group(
    gmx: str,
    build_dir: Path,
    replica_groups: dict[str, list[Path]],
    user_group: str | None,
) -> str:
    print_header("Detecting correct group")

    if not replica_groups:
        raise RuntimeError("No XTC files found")

    first_replica = sorted(
        replica_groups,
        key=replica_sort_key,
    )[0]

    first_xtc = replica_groups[first_replica][0]

    xtc_natoms = get_xtc_atom_count(gmx, first_xtc)
    print(f"[INFO] XTC atom count: {xtc_natoms}")

    index_file = build_dir / "index.ndx"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_file}"
        )

    group_sizes = parse_index_group_sizes(index_file)

    if user_group is not None:
        candidate_groups = [user_group]
    else:
        candidate_groups = [
            "Protein-H",
            "solute-H",
            "System",
        ]

    print("[INFO] Candidate group sizes:")

    for group_name in candidate_groups:
        size = group_sizes.get(group_name)

        if size is None:
            print(f"  {group_name}: not present")
        else:
            print(f"  {group_name}: {size}")

    matches = [
        group_name
        for group_name in candidate_groups
        if group_sizes.get(group_name) == xtc_natoms
    ]

    if len(matches) == 1:
        print(f"[AUTO] Using group: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        print(
            "[WARN] Multiple matching groups found: "
            + ", ".join(matches)
        )
        print(f"[AUTO] Using first match: {matches[0]}")
        return matches[0]

    candidate_text = ", ".join(
        f"{group_name}={group_sizes.get(group_name, 'missing')}"
        for group_name in candidate_groups
    )

    raise RuntimeError(
        f"No matching group found for XTC atom count {xtc_natoms}. "
        f"Candidate sizes: {candidate_text}"
    )


def clean_replica(
    gmx: str,
    pdb_code: str,
    replica_id: str,
    xtc_files: list[Path],
    dummy_tpr: Path,
    tmp_dir: Path,
) -> Path:
    print_header(f"Preparing replica {replica_id}")

    temporary_xtcs: list[Path] = []

    try:
        for index, xtc in enumerate(xtc_files):
            whole_xtc = (
                tmp_dir
                / f"{pdb_code}_{replica_id}_{index}_whole.xtc"
            )

            centred_xtc = (
                tmp_dir
                / f"{pdb_code}_{replica_id}_{index}_centred.xtc"
            )

            print(f"[INFO] Input: {xtc}")

            # dummy.tpr contains only the selected atoms.
            # In dummy.tpr, that selection is renamed to "System".

            # Step 1: make molecules whole.
            # This command asks only for the output group.
            run_cmd(
                [
                    gmx,
                    "trjconv",
                    "-s",
                    str(dummy_tpr),
                    "-f",
                    str(xtc),
                    "-o",
                    str(whole_xtc),
                    "-pbc",
                    "whole",
                ],
                input_text="System\n",
            )

            # Step 2: remove jumps and centre.
            # This asks for:
            #   1. centring group
            #   2. output group
            run_cmd(
                [
                    gmx,
                    "trjconv",
                    "-s",
                    str(dummy_tpr),
                    "-f",
                    str(whole_xtc),
                    "-o",
                    str(centred_xtc),
                    "-pbc",
                    "nojump",
                    "-center",
                ],
                input_text="System\nSystem\n",
            )

            temporary_xtcs.append(centred_xtc)
            whole_xtc.unlink(missing_ok=True)

        out_xtc = (
            tmp_dir
            / f"cleaned_{pdb_code}_{replica_id}.xtc"
        )

        out_xtc.unlink(missing_ok=True)

        run_cmd(
            [
                gmx,
                "trjcat",
                "-f",
                *[str(path) for path in temporary_xtcs],
                "-o",
                str(out_xtc),
                "-cat",
            ]
        )

    finally:
        for temporary_xtc in temporary_xtcs:
            temporary_xtc.unlink(missing_ok=True)

        for whole_xtc in tmp_dir.glob(
            f"{pdb_code}_{replica_id}_*_whole.xtc"
        ):
            whole_xtc.unlink(missing_ok=True)

    print(f"[OK] Cleaned trajectory: {out_xtc}")

    return out_xtc

# ---------------------------------------------------------------------
# XVG PARSING AND SUMMARY
# ---------------------------------------------------------------------

def parse_xvg(
    xvg_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    distances: list[float] = []

    if not xvg_path.exists():
        return np.array(times), np.array(distances)

    with xvg_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if (
                not line
                or line.startswith("@")
                or line.startswith("#")
            ):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                times.append(float(parts[0]))
                distances.append(float(parts[1]))
            except ValueError:
                continue

    return np.asarray(times), np.asarray(distances)


def summarise_distances(
    distances: np.ndarray,
) -> dict[str, float | int]:
    if len(distances) == 0:
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
        "n_frames": len(distances),
        "min_dist_nm": float(np.min(distances)),
        "max_dist_nm": float(np.max(distances)),
        "mean_dist_nm": float(np.mean(distances)),
        "median_dist_nm": float(np.median(distances)),
        "p01_dist_nm": float(np.percentile(distances, 1)),
        "p05_dist_nm": float(np.percentile(distances, 5)),
        "p95_dist_nm": float(np.percentile(distances, 95)),
        "p99_dist_nm": float(np.percentile(distances, 99)),
    }


def parse_complete_xvg_lines(
    data: bytes,
    threshold_nm: float,
) -> tuple[dict[str, float] | None, bytes]:
    lines = data.splitlines(keepends=True)
    incomplete_line = b""

    if lines and not lines[-1].endswith((b"\n", b"\r")):
        incomplete_line = lines.pop()

    for raw_line in lines:
        line = raw_line.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if (
            not line
            or line.startswith("@")
            or line.startswith("#")
        ):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            time_ps = float(parts[0])
            distance_nm = float(parts[1])
        except ValueError:
            continue

        if distance_nm <= threshold_nm:
            return {
                "time_ps": time_ps,
                "distance_nm": distance_nm,
            }, incomplete_line

    return None, incomplete_line


# ---------------------------------------------------------------------
# LIVE MINDIST MONITORING
# ---------------------------------------------------------------------

def terminate_process_group(
    process: subprocess.Popen,
    timeout_seconds: float = 10.0,
) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        process.wait()


def run_mindist_until_threshold(
    cmd: list[str],
    xvg_path: Path,
    log_path: Path,
    group_name: str,
    threshold_nm: float,
    poll_interval: float,
) -> dict[str, object]:
    xvg_path.unlink(missing_ok=True)

    threshold_reached = False
    threshold_time_ps: float | None = None
    threshold_distance_nm: float | None = None

    file_position = 0
    incomplete_data = b""

    with log_path.open("w") as log_fh:
        log_fh.write("COMMAND:\n")
        log_fh.write(" ".join(cmd) + "\n\n")
        log_fh.write(f"GROUP: {group_name}\n")
        log_fh.write(f"THRESHOLD_NM: {threshold_nm}\n")
        log_fh.write(
            f"POLL_INTERVAL_SECONDS: {poll_interval}\n\n"
        )
        log_fh.write("GROMACS OUTPUT:\n")
        log_fh.flush()

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        if process.stdin is None:
            terminate_process_group(process)
            raise RuntimeError(
                "Could not open stdin for gmx mindist"
            )

        try:
            # gmx mindist -pi normally asks for two groups.
            process.stdin.write(
                f"{group_name}\n{group_name}\n"
            )
            process.stdin.flush()
            process.stdin.close()

            while True:
                if xvg_path.exists():
                    with xvg_path.open("rb") as xvg_fh:
                        xvg_fh.seek(file_position)
                        new_data = xvg_fh.read()
                        file_position = xvg_fh.tell()

                    if new_data:
                        hit, incomplete_data = (
                            parse_complete_xvg_lines(
                                incomplete_data + new_data,
                                threshold_nm,
                            )
                        )

                        if hit is not None:
                            threshold_reached = True
                            threshold_time_ps = hit["time_ps"]
                            threshold_distance_nm = (
                                hit["distance_nm"]
                            )

                            print(
                                "\n[THRESHOLD] "
                                f"time={threshold_time_ps:.3f} ps | "
                                f"distance="
                                f"{threshold_distance_nm:.4f} nm | "
                                f"threshold={threshold_nm:.4f} nm"
                            )

                            terminate_process_group(process)
                            break

                returncode = process.poll()

                if returncode is not None:
                    # Read data written after the most recent poll.
                    if xvg_path.exists():
                        with xvg_path.open("rb") as xvg_fh:
                            xvg_fh.seek(file_position)
                            final_data = xvg_fh.read()

                        hit, _ = parse_complete_xvg_lines(
                            incomplete_data + final_data + b"\n",
                            threshold_nm,
                        )

                        if hit is not None:
                            threshold_reached = True
                            threshold_time_ps = hit["time_ps"]
                            threshold_distance_nm = (
                                hit["distance_nm"]
                            )

                    break

                time.sleep(poll_interval)

        except Exception:
            terminate_process_group(process)
            raise

        finally:
            if process.poll() is None:
                terminate_process_group(process)

        returncode = process.returncode

        log_fh.write("\n\nMONITOR RESULT:\n")
        log_fh.write(
            f"THRESHOLD_REACHED: {threshold_reached}\n"
        )
        log_fh.write(
            f"THRESHOLD_TIME_PS: {threshold_time_ps}\n"
        )
        log_fh.write(
            "THRESHOLD_DISTANCE_NM: "
            f"{threshold_distance_nm}\n"
        )
        log_fh.write(
            f"PROCESS_RETURNCODE: {returncode}\n"
        )

    # A SIGTERM return code is expected after a threshold hit.
    if not threshold_reached and returncode != 0:
        raise RuntimeError(
            f"gmx mindist failed with return code {returncode}. "
            f"See log: {log_path}"
        )

    return {
        "threshold_reached": threshold_reached,
        "time_ps": threshold_time_ps,
        "distance_nm": threshold_distance_nm,
        "returncode": returncode,
    }


def existing_xvg_threshold_result(
    xvg_path: Path,
    threshold_nm: float,
) -> dict[str, object]:
    times, distances = parse_xvg(xvg_path)

    matching_indices = np.flatnonzero(
        distances <= threshold_nm
    )

    if len(matching_indices) == 0:
        return {
            "threshold_reached": False,
            "time_ps": None,
            "distance_nm": None,
        }

    index = int(matching_indices[0])

    return {
        "threshold_reached": True,
        "time_ps": float(times[index]),
        "distance_nm": float(distances[index]),
    }


def write_summary(
    summary_csv: Path,
    summary_rows: list[dict[str, object]],
) -> None:
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
        "threshold_nm",
        "threshold_time_ps",
        "threshold_distance_nm",
        "status",
    ]

    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare trajectories, calculate periodic-image distances, "
            "and stop when the selected threshold is reached."
        )
    )

    parser.add_argument(
        "--pdb",
        required=True,
    )

    parser.add_argument(
        "--gmx",
        default=GMX_DEFAULT,
    )

    parser.add_argument(
        "--group",
        choices=[
            "Protein-H",
            "solute-H",
            "System",
        ],
        default=None,
        help=(
            "Force a specific index group. By default, the group is "
            "selected by matching its atom count to the XTC."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help=(
            "Stop when the periodic-image distance is less than or "
            "equal to this value in nm. Default: 1.2."
        ),
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help=(
            "Seconds between checks of the live XVG output. "
            "Default: 0.2."
        ),
    )

    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all files in the temporary directory first.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Use existing cleaned XTC and XVG files where possible."
        ),
    )

    args = parser.parse_args()

    if args.threshold <= 0:
        parser.error("--threshold must be greater than zero")

    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than zero")

    pdb_code = args.pdb.lower()
    pdb_dir = get_pdb_dir(pdb_code)
    build_dir = pdb_dir / "build"

    cwd = Path.cwd()

    out_root = cwd / "results" / pdb_code
    tmp_dir = out_root / "tmp2"
    dist_dir = out_root / "pi_dist"

    dummy_tpr = tmp_dir / "dummy.tpr"
    summary_csv = (
        dist_dir
        / f"pi_dist_summary_{pdb_code}.csv"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    print_header("Paths")
    print(f"Working dir: {cwd}")
    print(f"PDB dir:     {pdb_dir}")
    print(f"Build dir:   {build_dir}")
    print(f"TMP dir:     {tmp_dir}")
    print(f"Output dir:  {dist_dir}")
    print(f"GROMACS:     {args.gmx}")
    print(f"Threshold:   {args.threshold:.4f} nm")

    # --------------------------------------------------------------
    # Step 1: collect and group original trajectories
    # --------------------------------------------------------------

    xtc_paths = collect_xtc_paths(pdb_dir)

    if not xtc_paths:
        raise FileNotFoundError(
            f"No input XTC trajectories found under {pdb_dir}"
        )

    replica_groups = group_xtcs_by_replica(xtc_paths)

    print(f"Replicas:     {len(replica_groups)}")
    print(f"XTC files:    {len(xtc_paths)}")

    # --------------------------------------------------------------
    # Step 2: detect group and create dummy TPR
    # --------------------------------------------------------------

    group_name = detect_matching_group(
        args.gmx,
        build_dir,
        replica_groups,
        args.group,
    )

    print(f"Group:        {group_name}")

    if args.skip_existing and dummy_tpr.exists():
        print(
            f"[SKIP] Using existing dummy TPR: {dummy_tpr}"
        )
    else:
        cleanup_tmp_dir(
            tmp_dir,
            pdb_code,
            wipe=args.wipe,
        )

        make_dummy_tpr(
            args.gmx,
            build_dir,
            dummy_tpr,
            group_name,
        )

    # --------------------------------------------------------------
    # Step 3: clean each replica, then run mindist immediately
    # --------------------------------------------------------------

    summary_rows: list[dict[str, object]] = []
    threshold_found = False

    for replica_id in sorted(
        replica_groups,
        key=replica_sort_key,
    ):
        cleaned_xtc = (
            tmp_dir
            / f"cleaned_{pdb_code}_{replica_id}.xtc"
        )

        xvg_out = (
            dist_dir
            / f"pi_dist_{pdb_code}_{replica_id}.xvg"
        )

        log_out = (
            dist_dir
            / f"pi_dist_{pdb_code}_{replica_id}.log"
        )

        print_header(f"Replica {replica_id}")

        # Prepare the current replica only when required.
        if (
            args.skip_existing
            and cleaned_xtc.exists()
        ):
            print(
                "[SKIP] Using existing cleaned trajectory: "
                f"{cleaned_xtc.name}"
            )
        else:
            cleaned_xtc = clean_replica(
                args.gmx,
                pdb_code,
                replica_id,
                replica_groups[replica_id],
                dummy_tpr,
                tmp_dir,
            )
        # Reuse an existing XVG if requested.
        if args.skip_existing and xvg_out.exists():
            print(
                f"[SKIP] Using existing XVG: {xvg_out.name}"
            )

            threshold_result = (
                existing_xvg_threshold_result(
                    xvg_out,
                    args.threshold,
                )
            )

            result = {
                **threshold_result,
                "returncode": None,
            }

            status = (
                "threshold_reached_existing"
                if result["threshold_reached"]
                else "skipped_existing"
            )

        else:
            cmd = [
                args.gmx,
                "mindist",
                "-s",
                str(dummy_tpr),
                "-f",
                str(cleaned_xtc),
                "-pi",
                "-od",
                str(xvg_out),
            ]

            result = run_mindist_until_threshold(
                cmd=cmd,
                xvg_path=xvg_out,
                log_path=log_out,
                group_name="System",
                threshold_nm=args.threshold,
                poll_interval=args.poll_interval,
            )

            status = (
                "threshold_reached"
                if result["threshold_reached"]
                else "completed"
            )

        _, distances = parse_xvg(xvg_out)
        stats = summarise_distances(distances)

        summary_rows.append(
            {
                "pdb": pdb_code,
                "replica": replica_id,
                "xtc": cleaned_xtc.name,
                "xvg": xvg_out.name,
                **stats,
                "threshold_nm": args.threshold,
                "threshold_time_ps": result["time_ps"],
                "threshold_distance_nm": (
                    result["distance_nm"]
                ),
                "status": status,
            }
        )

        # Write after every replica so results survive interruption.
        write_summary(summary_csv, summary_rows)

        if result["threshold_reached"]:
            threshold_found = True

            print(
                f"[STOP] Replica {replica_id} reached the "
                f"threshold at {result['time_ps']:.3f} ps: "
                f"{result['distance_nm']:.4f} nm"
            )

            break

        print(
            f"[OK] min={stats['min_dist_nm']:.4f} nm | "
            f"p05={stats['p05_dist_nm']:.4f} nm | "
            f"mean={stats['mean_dist_nm']:.4f} nm"
        )

    print_header("DONE")

    if threshold_found:
        print(
            "The calculation stopped after detecting a periodic-image "
            f"distance <= {args.threshold:.4f} nm."
        )
    else:
        print(
            "All available replicas completed without reaching "
            f"{args.threshold:.4f} nm."
        )

    print(f"Summary CSV: {summary_csv}")
    print(f"Distance output: {dist_dir}")


if __name__ == "__main__":
    main()