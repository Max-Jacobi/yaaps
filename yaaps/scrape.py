"""
Directory scraper for GRAthena++ athdf output.

This module scrapes a simulation output directory for ``*.athdf`` dump
files and exposes lookups (variable metadata, grids, field data, time <->
iteration mapping) needed by :mod:`yaaps.datatypes`.

Unlike a naive scraper, this implementation never assumes iteration
numbers are contiguous: only iterations that actually have a file on disk
are ever reported, so deleting dump files (e.g. every second one, to save
disk space) is safe. A small on-disk JSON index cache is kept per
simulation directory so that repeated process invocations don't have to
re-open every dump file's HDF5 header to read its ``Time`` attribute.

A simulation directory may also be structured as a sequence of restart
directories (``output-0000``, ``output-0001``, ...), as produced by
GRAthena++ when a run is resumed from a checkpoint. Restarts can overlap:
a later restart may rewrite iterations already dumped by an earlier one
(with corrected/continued data). This module resolves that transparently
- for a given ("out" set, iteration) pair found in more than one restart
directory, the file with the newer mtime wins - so no external "combine"
step (symlink farm + ascii concatenation) is needed beforehand.
"""

import os
import re
import json
import bisect
import warnings
from functools import lru_cache
from typing import Union

import numpy as np
import h5py


class IterationNotAvailable(LookupError):
    """Raised when a requested iteration has no corresponding file on disk."""


_FN_RE = re.compile(r"^(?P<stem>.+)\.(?P<out>out\d+)\.(?P<iter>\d+)\.athdf$")
_CACHE_BASENAME = ".yaaps_scrape_cache.json"
_CACHE_VERSION = 1
_RESTART_PREFIX = "output-"


def find_restart_dirs(path: str) -> list[str]:
    """
    Return the ordered list of directories holding a simulation's raw output.

    If `path` itself contains ``output-*`` restart subdirectories (as
    produced by GRAthena++ when a run is resumed from a checkpoint), returns
    those subdirectories in ascending order. Otherwise returns ``[path]``
    unchanged - a single flat output directory, a single restart directory,
    or an already-combined directory are all handled identically by callers.
    """
    try:
        subdirs = sorted(
            entry.name for entry in os.scandir(path)
            if entry.is_dir() and entry.name.startswith(_RESTART_PREFIX)
        )
    except OSError:
        subdirs = []
    if subdirs:
        return [os.path.join(path, name) for name in subdirs]
    return [path]


def _decode_attr(values) -> tuple:
    """Decode an HDF5 attribute array of (possibly bytes) strings to str."""
    return tuple(v.decode("ascii") if isinstance(v, bytes) else v for v in values)


def _split_chunks(flat: list, sizes: list) -> list:
    """Split a flat list into sublists of the given sizes."""
    chunks = []
    i = 0
    for size in sizes:
        chunks.append(list(flat[i:i + size]))
        i += size
    return chunks


class AthdfScraper:
    """
    Scrape a directory of GRAthena++ athdf output files.

    Groups dump files by "outN" set, indexes which iterations actually
    exist on disk (never assuming contiguity), and provides lookups for
    variable metadata, grids and field data.

    Args:
        dir_data: Path to the simulation output directory.
        N_B: Meshblock size (nx1, nx2, nx3), used to infer ghost zone counts.

    Raises:
        RuntimeError: If no readable athdf files are found in the directory.
    """

    def __init__(self, dir_data: str, N_B: Union[int, float, list, tuple]):
        self.dir_data = os.path.abspath(dir_data)
        if isinstance(N_B, (int, float)):
            N_B = (N_B, N_B, N_B)
        self.N_B = tuple(int(v) for v in N_B)

        self._stems: dict[str, str] = {}
        self._fns: dict[str, dict[int, str]] = {}
        self._times: dict[str, dict[int, float]] = {}
        self._iters_arr: dict[str, np.ndarray] = {}
        self._times_arr: dict[str, np.ndarray] = {}
        self._out_dataset_names: dict[str, tuple] = {}
        self._out_variable_names: dict[str, tuple] = {}
        self._out_num_variables: dict[str, tuple] = {}
        self._out_ng: dict[str, tuple] = {}

        self._scan()
        self._var_map = self._parse_to_var_key()

    # -- scanning / caching --------------------------------------------

    def _scan(self) -> None:
        source_dirs = find_restart_dirs(self.dir_data)

        # out -> {iter: (name, abspath, mtime, size)}; when the same (out,
        # iter) pair appears in more than one restart directory (a later
        # restart rewriting an iteration an earlier one already dumped),
        # keep whichever has the newer mtime - mirrors the external
        # `combine` script's merge_best_maps behavior.
        raw: dict[str, dict[int, tuple[str, str, float, int]]] = {}
        for d in source_dirs:
            try:
                entries = list(os.scandir(d))
            except OSError as exc:
                raise RuntimeError(f"Cannot list directory {d}: {exc}") from exc
            for entry in entries:
                if not entry.name.endswith(".athdf"):
                    continue
                m = _FN_RE.match(entry.name)
                if m is None or not entry.is_file():
                    continue
                st = entry.stat()
                out = m.group("out")
                it = int(m.group("iter"))
                self._stems.setdefault(out, m.group("stem"))
                bucket = raw.setdefault(out, {})
                prev = bucket.get(it)
                if prev is None or st.st_mtime > prev[2]:
                    bucket[it] = (entry.name, entry.path, st.st_mtime, st.st_size)

        if not raw:
            raise RuntimeError(f"No athdf files found under {self.dir_data}")

        sorted_raw = {
            out: sorted(
                ((it, *info) for it, info in bucket.items()), key=lambda item: item[0]
            )
            for out, bucket in raw.items()
        }

        cache = self._load_cache()
        cached_out_sets = cache["out_sets"] if cache is not None else {}
        new_cache_out_sets = {}

        for out, files in sorted_raw.items():
            cached_iters = cached_out_sets.get(out, {}).get("iterations", {})
            fns: dict[int, str] = {}
            times: dict[int, float] = {}
            cache_entries: dict[str, dict] = {}

            n_files = len(files)
            for i, (it, name, fn_abs, mtime, size) in enumerate(files):
                is_last = i == n_files - 1
                rel = os.path.relpath(fn_abs, self.dir_data)
                cached_entry = cached_iters.get(str(it))
                reuse = (
                    not is_last
                    and cached_entry is not None
                    and cached_entry.get("filename") == rel
                    and cached_entry.get("mtime") == mtime
                    and cached_entry.get("size") == size
                )
                if reuse:
                    time_val = cached_entry["time"]
                else:
                    try:
                        with h5py.File(fn_abs, "r") as f:
                            time_val = float(f.attrs["Time"])
                    except OSError:
                        if not is_last:
                            warnings.warn(
                                f"Could not read Time from {fn_abs}; skipping.",
                                RuntimeWarning,
                            )
                        # last file of an out-set being unreadable means the
                        # simulation is still actively writing it - skip
                        # silently, it will be picked up on a later scan.
                        continue

                fns[it] = fn_abs
                times[it] = time_val
                cache_entries[str(it)] = {
                    "filename": rel, "mtime": mtime, "size": size, "time": time_val,
                }

            if not fns:
                continue

            iters_sorted = sorted(fns)
            self._fns[out] = fns
            self._times[out] = times
            self._iters_arr[out] = np.array(iters_sorted, dtype=np.int64)
            self._times_arr[out] = np.array([times[it] for it in iters_sorted], dtype=np.float64)
            new_cache_out_sets[out] = {"stem": self._stems[out], "iterations": cache_entries}

        if not self._fns:
            raise RuntimeError(f"No readable athdf files found under {self.dir_data}")

        self._save_cache({
            "version": _CACHE_VERSION,
            "N_B": list(self.N_B),
            "out_sets": new_cache_out_sets,
        })

        for out in self._fns:
            dataset_names, variable_names, num_variables = self._read_out_attrs(out)
            self._out_dataset_names[out] = dataset_names
            self._out_variable_names[out] = variable_names
            self._out_num_variables[out] = num_variables

    def _cache_path(self) -> str:
        return os.path.join(self.dir_data, _CACHE_BASENAME)

    def _load_cache(self) -> Union[dict, None]:
        try:
            with open(self._cache_path(), "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("version") != _CACHE_VERSION:
            return None
        if list(data.get("N_B", [])) != list(self.N_B):
            return None
        return data

    def _save_cache(self, data: dict) -> None:
        path = self._cache_path()
        tmp = f"{path}.tmp{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    # -- HDF5 shape/metadata introspection -------------------------------

    def _read_out_attrs(self, out: str) -> tuple:
        # DatasetNames/VariableNames/NumVariables are constant across an
        # out-set's dumps, so only the first (lowest-iteration) file is read.
        it0 = int(self._iters_arr[out][0])
        fn = self._fns[out][it0]
        with h5py.File(fn, "r") as f:
            dataset_names = _decode_attr(f.attrs["DatasetNames"])
            variable_names = _decode_attr(f.attrs["VariableNames"])
            num_variables = tuple(int(v) for v in f.attrs["NumVariables"])
        return dataset_names, variable_names, num_variables

    def _get_data_spec_iter(self, out: str, iterate: int) -> tuple:
        fn = self.get_iter_fn(out, iterate)
        dataset_name = self._out_dataset_names[out][0]
        with h5py.File(fn, "r") as f:
            shape = f[dataset_name].shape
        num_vars, num_meshblocks = shape[0], shape[1]
        dim_xyz = shape[-3:][::-1]
        slicing = tuple(f"x{ix}" for ix in range(1, 4) if dim_xyz[ix - 1] > 1)
        return num_vars, num_meshblocks, dim_xyz, slicing

    def _get_ghosts(self, dim_xyz: tuple, N_B: tuple) -> tuple:
        dg = []
        have_ghosts = True
        for d, N in zip(dim_xyz, N_B):
            if d > 1:
                cur_dg = abs(d - N) // 2
                have_ghosts = have_ghosts and (cur_dg > 1)
                dg.append(cur_dg)
            else:
                dg.append(0)
        return have_ghosts, max(dg)

    def _parse_to_var_key(self) -> dict:
        var_map = {}
        for out in self._fns:
            maxiter = int(self._iters_arr[out][-1])
            _, _, dim_xyz, slicing = self._get_data_spec_iter(out, maxiter)

            dataset_names = self._out_dataset_names[out]
            vars_split = _split_chunks(
                self._out_variable_names[out], self._out_num_variables[out]
            )
            have_ghosts, dg = self._get_ghosts(dim_xyz, self.N_B)
            self._out_ng[out] = (dim_xyz, dg)

            for var in self._out_variable_names[out]:
                var_index = dataset_index = None
                for vix, vs in enumerate(vars_split):
                    if var in vs:
                        var_index = vs.index(var)
                        dataset_index = vix
                        break
                dataset_name = dataset_names[dataset_index]
                var_map[(var, slicing, have_ghosts)] = (out, var_index, dg, dataset_name)
        return var_map

    @lru_cache(maxsize=None, typed=True)
    def _parse_var_info(self, var: str, sampling, with_ghosts: bool = True) -> tuple:
        if isinstance(sampling, str):
            sampling = (sampling,)
        slicing = tuple(val[:2] for val in sampling)
        var_key = (var, slicing, with_ghosts)
        try:
            var_info = self._var_map[var_key]
        except KeyError:
            raise KeyError(
                f"Could not find data corresponding to key: {var_key}.\n"
                f"Known keys (var, slicing, with_ghosts): {list(self._var_map.keys())}"
            ) from None
        return sampling, var_info

    # -- public API --------------------------------------------------------

    def debug_data_keys(self) -> dict:
        """Return the ``(var, slicing, has_ghosts) -> (out, var_index, dg, dataset_name)`` map."""
        return self._var_map

    def get_var_info(self, var: str, sampling, with_ghosts: bool = True) -> tuple:
        """Given a variable and sampling, return ``(out, var_index, dg, dataset_name)``."""
        return self._parse_var_info(var, sampling, with_ghosts)[1]

    def get_available_iters(self, var: str, sampling, with_ghosts: bool = True) -> np.ndarray:
        """Return the sorted array of iterations that actually exist on disk."""
        out = self._parse_var_info(var, sampling, with_ghosts)[1][0]
        return self._iters_arr[out].copy()

    def get_available_times(self, var: str, sampling, with_ghosts: bool = True) -> np.ndarray:
        """Return simulation times aligned 1:1 with :meth:`get_available_iters`."""
        out = self._parse_var_info(var, sampling, with_ghosts)[1][0]
        return self._times_arr[out].copy()

    def get_iter_time(self, out: str, iterate: int) -> float:
        """Look up the simulation time of a given iteration (O(1), no file I/O)."""
        try:
            return self._times[out][int(iterate)]
        except KeyError:
            raise IterationNotAvailable(
                f"No cached Time for out={out!r} iterate={iterate} in {self.dir_data!r}."
            ) from None

    def get_iter_from_time(self, var: str, sampling, time: float, with_ghosts: bool = True) -> int:
        """Snap a requested time to the nearest iteration that actually exists on disk."""
        out = self._parse_var_info(var, sampling, with_ghosts)[1][0]
        iters, times = self._iters_arr[out], self._times_arr[out]
        idx = np.searchsorted(times, time)
        if idx <= 0:
            return int(iters[0])
        if idx >= len(times):
            return int(iters[-1])
        lo_diff = time - times[idx - 1]
        hi_diff = times[idx] - time
        return int(iters[idx - 1] if lo_diff <= hi_diff else iters[idx])

    def get_iter_fn(self, out: str, iterate: int) -> str:
        """Return the file path for a given out-set and iteration.

        Raises:
            IterationNotAvailable: If no file exists for that iteration
                (e.g. it was deleted), listing nearby available iterations.
        """
        try:
            return self._fns[out][int(iterate)]
        except KeyError:
            available = sorted(self._fns.get(out, {}))
            pos = bisect.bisect_left(available, int(iterate))
            nearby = available[max(0, pos - 3):pos + 3]
            raise IterationNotAvailable(
                f"Iteration {iterate} not found for out={out!r} in {self.dir_data!r}. "
                f"Nearby available iterations: {nearby}"
            ) from None

    @lru_cache(maxsize=None, typed=True)
    def slicer_physical(self, ng: int, sampling, dg=None) -> tuple:
        """Given a ghost zone count and sampling, build slicers to physical nodes."""
        var_sl = []
        fcn_sl = [slice(None)]
        if ng > 0:
            for gix, grid_var in enumerate(sampling):
                offset = 1 if "f" in grid_var else 0
                il, iu = ng, self.N_B[gix] + ng + offset
                if dg is not None:
                    il, iu = il - dg, iu + dg
                nog_sl = slice(il, iu, None)
                var_sl.append((slice(None), nog_sl))
                fcn_sl.append(nog_sl)
        else:
            for _ in sampling:
                var_sl.append((slice(None), slice(None)))
                fcn_sl.append(slice(None))
        return tuple(var_sl), tuple(fcn_sl)

    def get_grid(self, out: str, sampling, iterate: int = 0, strip_dg: int = 0) -> tuple:
        """Extract grid coordinate arrays for the given out-set/sampling/iteration."""
        sampling = tuple(sampling)
        fn_data = self.get_iter_fn(out, iterate)
        with h5py.File(fn_data, "r") as f:
            xyz = [np.array(f[grid_var]) for grid_var in sampling]
        if strip_dg > 0:
            _, ng = self._out_ng[out]
            var_sl, _ = self.slicer_physical(ng, sampling, dg=ng - strip_dg)
            xyz = tuple(gr[sl] for gr, sl in zip(xyz, var_sl))
        else:
            xyz = tuple(xyz)
        return xyz

    def get_grid_levels(self, out: str, iterate: int = 0) -> np.ndarray:
        """Return the refinement levels each MeshBlock lives on."""
        fn_data = self.get_iter_fn(out, iterate)
        with h5py.File(fn_data, "r") as f:
            return np.array(f["Levels"])

    def get_var(
        self, var: str, sampling, iterate: int = 0,
        with_ghosts: bool = True, strip_dg: int = 0,
    ) -> np.ndarray:
        """Extract field data for a variable at a given iteration."""
        sampling, var_info = self._parse_var_info(var, sampling, with_ghosts)
        out, var_index, dg, dataset_name = var_info

        fn_data = self.get_iter_fn(out, iterate)
        with h5py.File(fn_data, "r") as f:
            dataset = np.array(f[dataset_name])

        field_data = np.transpose(dataset[var_index], (0, 3, 2, 1)).squeeze()

        if strip_dg > 0:
            _, ng = self._out_ng[out]
            _, fcn_sl = self.slicer_physical(ng, sampling, dg=ng - strip_dg)
            try:
                field_data = field_data[fcn_sl]
            except IndexError:
                # single-MeshBlock dataset: re-add the stripped block axis
                field_data = field_data[fcn_sl[1:]][None, ...]

        return field_data
