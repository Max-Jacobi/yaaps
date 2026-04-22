#!/usr/bin/env python3
"""
combine - efficient, readable Python implementation using multiprocessing.

Usage:
    ./combine /path/to/sim [num_link_workers] [num_scan_workers]

- num_link_workers: number of threads used to create symlinks (default: os.cpu_count()).
- num_scan_workers: number of processes used to scan restart directories (default: min(num_link_workers, len(restarts))).

Behavior summary:
- Determine restarts (output-*) and whether we're running incrementally via combine/.last_combine.
- If incremental: only process restarts that contain any file newer than the timestamp file (fast top-level check).
- On fresh run: copy header lines and .par/.inp from the first restart that contains them.
- Use a process pool to scan each restart directory in parallel; each worker returns the best (latest mtime) candidate per basename for '*.athdf*' and '*.surface*'.
- Merge worker results to a global best-per-basename map.
- Create atomic symlinks in parallel (thread pool) so each link points to the newest file.
- Append ASCII files (.hst, tra.*, wav_*, *.horizon_summary*) sequentially.
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Iterable
import concurrent.futures
import multiprocessing
import random
import string

# -------------------------
# Helpers / small pure funcs
# -------------------------

def parse_args() -> tuple[Path, int, int, bool]:
    p = argparse.ArgumentParser(description="Combine simulation restart outputs")
    p.add_argument("sim", type=Path, help="Path to simulation directory")
    p.add_argument("num_link_workers", type=int, nargs="?", default=4,
                   help="Number of parallel threads for creating links (default: CPU count)")
    p.add_argument("num_scan_workers", type=int, nargs="?", default=0,
                   help="Number of parallel processes to scan directories (default: min(num_link_workers, #restarts))")
    p.add_argument("-f", '--force', action='store_true',
                   help="Force fresh combine (ignore existing timestamp)")
    args = p.parse_args()
    sim = args.sim.resolve()
    if not sim.exists() or not sim.is_dir():
        p.error(f"{sim} is not a directory")
    return sim, args.num_link_workers, args.num_scan_workers, args.force

def list_restarts(sim: Path) -> list[str]:
    return sorted([p.name for p in sim.glob("output-*") if p.is_dir()])

def restart_has_newer_file(restart_dir: Path, ts_path: Path) -> bool:
    """Fast check: any regular file in restart_dir (non-recursive) with mtime > ts_file?"""
    if not ts_path.exists():
        return True
    ts_mtime = ts_path.stat().st_mtime
    try:
        with os.scandir(restart_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime > ts_mtime:
                        return True
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return False
    return False

def copy_head_lines(src: Path, dst: Path, lines: int) -> None:
    try:
        with src.open("rb") as sf, dst.open("wb") as df:
            for _ in range(lines):
                line = sf.readline()
                if not line:
                    break
                df.write(line)
    except FileNotFoundError:
        pass

def copy_headers_from_first(restarts: Iterable[str], sim: Path, combine: Path) -> None:
    for rst in restarts:
        rd = sim / rst
        if not rd.is_dir():
            continue
        hst = next(rd.glob("*.hst"), None)
        if hst:
            copy_head_lines(hst, combine / hst.name, 2)
        for tra in rd.glob("tra.*"):
            copy_head_lines(tra, combine / tra.name, 1)
        for wav in rd.glob("wav_*"):
            copy_head_lines(wav, combine / wav.name, 1)
        for hor in rd.glob("*.horizon_summary*"):
            copy_head_lines(hor, combine / hor.name, 1)
        for par in rd.glob("*.par"):
            try:
                shutil.copy2(par, combine / par.name)
            except FileNotFoundError:
                pass
        for inp in rd.glob("*.inp"):
            try:
                shutil.copy2(inp, combine / inp.name)
            except FileNotFoundError:
                pass
        break

# -------------------------
# Scanning (multiprocessing)
# -------------------------

def scan_restart_for_best(rd: str) -> dict[str, tuple[float, str]]:
    """
    Run in worker process: scan rd (string path) and return mapping:
    basename -> (mtime_seconds, absolute_path_str)
    Only consider files where name contains 'athdf' or 'surface'.
    Non-recursive. Uses os.scandir for efficiency.
    """
    file_times: dict[str, tuple[float, str]] = {}
    with os.scandir(rd) as it:
        for entry in it:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except FileNotFoundError:
                continue
            name = entry.name
            if ("athdf" not in name) and ("surface" not in name):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            mtime = st.st_mtime
            file_times[name] = (mtime, os.path.abspath(entry.path))
    return file_times

def merge_best_maps(maps: Iterable[dict[str, tuple[float, str]]]) -> dict[str, str]:
    """
    Merge multiple worker maps into basename -> absolute_path of the newest mtime.
    """
    global_best: dict[str, tuple[float, str]] = {}
    for mp in maps:
        for name, (mtime, path) in mp.items():
            prev = global_best.get(name)
            if (prev is None) or (mtime > prev[0]):
                global_best[name] = (mtime, path)
    return {name: path for name, (_, path) in global_best.items()}

# -------------------------
# Linking (threaded)
# -------------------------

def _random_suffix(n: int = 6) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def atomic_symlink(src: str, dst: Path) -> None:
    """
    Simple atomic symlink:
      - create a temporary symlink next to dst
      - os.replace it to dst (atomic)
    Let exceptions propagate to caller (caller will handle/log).
    """
    tmp = dst.parent / (f".{dst.name}.tmp.{os.getpid()}.{_random_suffix()}")
    os.symlink(src, str(tmp))
    os.replace(str(tmp), str(dst))

def create_link_worker(src_path: str, combine_dir: Path):
    dst = combine_dir / os.path.basename(src_path)
    try:
        # replace src_path (absolute) with a relative path
        src_path = os.path.relpath(src_path, start=combine_dir)
        atomic_symlink(src_path, dst)
    except Exception as e:
        return dst.name, False, str(e)
    return dst.name, True, "linked"

def create_links_parallel(paths: Iterable[str], combine_dir: Path, num_workers: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(create_link_worker, p, combine_dir) for p in paths]
        for fut in concurrent.futures.as_completed(futures):
            name, ok, msg = fut.result()
            if not ok:
                print(f"ERROR linking {name}: {msg}", file=sys.stderr)

# -------------------------
# ASCII appending (sequential)
# -------------------------

def append_tail_lines(src: Path, dst: Path, skip: int) -> None:
    try:
        with src.open("rb") as sf, dst.open("ab") as df:
            for _ in range(skip):
                ln = sf.readline()
                if not ln:
                    return
            while True:
                chunk = sf.read(64 * 1024)
                if not chunk:
                    break
                df.write(chunk)
    except FileNotFoundError:
        pass

def append_ascii_from_restart(rd: Path, combine: Path) -> None:
    for hst in rd.glob("*.hst"):
        append_tail_lines(hst, combine / hst.name, 2)
    for tra in rd.glob("tra.*"):
        append_tail_lines(tra, combine / tra.name, 1)
    for wav in rd.glob("wav_*"):
        append_tail_lines(wav, combine / wav.name, 1)
    for hor in rd.glob("*.horizon_summary*"):
        append_tail_lines(hor, combine / hor.name, 1)

# -------------------------
# Main flow
# -------------------------

def main() -> int:
    sim, num_link_workers, num_scan_workers, force_fresh = parse_args()
    print(f"{sim.name}: ", end="")
    restarts = list_restarts(sim)
    if not restarts:
        print("No restart directories found.")
        return 0

    combine = sim / "combine"
    combine.mkdir(parents=True, exist_ok=True)
    timestamp = combine / ".last_combine"
    incremental = timestamp.exists() and not force_fresh

    # if incremental:
    #     print(f"Incremental combine: processing only restarts with files newer than {timestamp} (mtime={timestamp.stat().st_mtime})")
    # else:
    #     print("Fresh combine: processing all restarts")

    restarts_to_process: list[Path] = []
    if incremental:
        for rst in restarts:
            rd = sim / rst
            if restart_has_newer_file(rd, timestamp):
                restarts_to_process.append(rd)
            # else:
            #     print(f"Skipping {rst} (no files newer than last combine)")
    else:
        restarts_to_process = [sim / r for r in restarts]

    if not restarts_to_process:
        print("Nothing to do.")
        timestamp.parent.mkdir(parents=True, exist_ok=True)
        timestamp.touch()
        return 0

    if not incremental:
        copy_headers_from_first(restarts, sim, combine)

    if num_scan_workers <= 0:
        num_scan_workers = min(num_link_workers, max(1, len(restarts_to_process)))
    num_scan_workers = max(1, num_scan_workers)

    # print(f"Processing {len(restarts_to_process)} restart(s). scan_workers={num_scan_workers}, link_workers={num_link_workers}")

    scan_inputs = [str(p) for p in restarts_to_process]
    maps: list[dict[str, tuple[float, str]]] = []
    if num_scan_workers == 1:
        for rd_str in scan_inputs:
            maps.append(scan_restart_for_best(rd_str))
    else:
        with multiprocessing.Pool(processes=num_scan_workers) as pool:
            for res in pool.imap_unordered(scan_restart_for_best, scan_inputs):
                maps.append(res)

    best_map = merge_best_maps(maps)
    if not best_map:
        print("No athdf/surface files found to link.")
    else:
        create_links_parallel(list(best_map.values()), combine, num_link_workers)

    for rd in restarts_to_process:
        # print(f"Appending ascii files for {rd.name}")
        append_ascii_from_restart(rd, combine)

    timestamp.parent.mkdir(parents=True, exist_ok=True)
    timestamp.touch()
    print(f"Done. Combined {len(restarts_to_process)} restart(s).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
