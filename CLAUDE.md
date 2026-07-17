# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

YAAPS (Yet Another Athena Plot Script) is a Python package for post-processing and plotting output from GRAthena++ (general-relativistic Athena++) neutron star merger simulations. Requires Python ≥ 3.12 (uses PEP 695 generics like `class ColorPlot[DataType: MeshData]`).

## Commands

```bash
pip install -e ".[dev]"     # editable install (deps: numpy, scipy, h5py, matplotlib, tqdm)
pytest                      # run tests
pytest tests/test_scrape.py::TestGapHandling::test_missing_middle_iteration  # single test
ruff check yaaps tests examples   # lint (config in pyproject.toml)
ruff format yaaps tests examples  # format (Black-compatible style)
```

Tests in `tests/test_scrape.py` are skipped entirely unless a local fixture `.athdf` file exists (hardcoded path in the `FIXTURE` constant); they generate test data by copying and mutating that file.

Lint/format is Ruff, configured in `pyproject.toml`; run both `ruff check` and `ruff format` before committing. `yaaps/athena_read.py` is excluded — it is vendored Athena++ code and must stay diffable against upstream, so don't reformat or restyle it. Type checking is done by ty (the user runs `ty-ls` in their editor); there is no type-check step in the repo itself.

Batch plotting CLI:

```bash
python -m yaaps.batch_plots --sims /path/sim1 /path/sim2 --output-dir ./plots --config plots.toml --cpus 8
```

## Architecture

Data flows in layers: `Simulation` (entry point) → `AthdfScraper` (file discovery/HDF5 reads) → `MeshData` subclasses (loading/interpolation) → `Plot` subclasses (matplotlib rendering).

- **`simulation.py` — `Simulation`**: the single user-facing entry point (only export of `yaaps/__init__.py`). Finds the `.inp`/`.par` parameter file, exposes `.hst` (history), `.wav()` (waveforms), `.tra()` (tracers), `.horizon()`, `.scrape`, and convenience `plot2d()`/`animate2d()`. `complete_var()` resolves abbreviated variable names/aliases (e.g. `"rho"` → `"hydro.prim.rho"`) against what's actually on disk.

- **`scrape.py` — `AthdfScraper`**: scans a simulation directory for `*.athdf` dumps and maps (variable, sampling, ghosts) → files, plus time ↔ iteration lookups. Two invariants it is built around:
  - **Gap-aware**: iteration numbers are never assumed contiguous; only files actually on disk are reported (users delete dumps to save space). Raises `IterationNotAvailable` otherwise.
  - **Restart merging**: a run resumed from checkpoints produces `output-0000`, `output-0001`, ... subdirectories that may overlap. `find_restart_dirs()` + newest-mtime-wins resolution merge them transparently — no external "combine" step. ASCII files (.hst, wav, tra) are likewise concatenated across restarts in `simulation.py` (`_load_ascii_multi`), keeping only the first header, and deduplicated/sorted by iter/time (`_straighten`).
  - Keeps an on-disk JSON index cache (`.yaaps_scrape_cache.json`) per directory so repeat invocations don't re-open every HDF5 header. Filesystem calls are deliberately minimized throughout (target: slow network filesystems) — preserve this when editing.

- **`datatypes.py` — `MeshData` hierarchy**: `Native` (variable read directly from file), `Derived` (computed from Native dependencies via a user callable; the callable may optionally accept `xyz`, `time`, `sampling` kwargs, detected by signature inspection), `Vector` (two components for quiver/stream plots). All operate on per-meshblock arrays (axis 0 = meshblock index) and provide interpolation onto regular grids. Sampling is a plane spec like `('x1v', 'x2v')` or shorthand `"xy"`.

- **`plot2D.py` — `Plot` hierarchy**: `NativeColorPlot`/`DerivedColorPlot`, contour, quiver, stream, tracer-scatter, meshblock-overlay classes, plus `animate()` and `save_frames()` (parallel frame rendering via `parallel_utils.do_parallel`). Each plot holds a `MeshData` in `.data` and re-renders per time for animations.

- **Presentation layer**: `decorations.py` (per-variable default cmap/norm and `var_alias` map), `plot_formatter.py` (`PlotFormatter(mode)` factory: `"raw"` = code units/names, `"paper"` = LaTeX labels + physical units), `units.py` (code→CGS conversion factors, matched by endswith or regex), `recipes2D.py` (metric-tensor helpers for derived GR quantities).

- **`batch_plots.py`**: standalone TOML-driven CLI producing hst plots, sim/var-combined grids, and their animation variants across many simulations with multiprocessing. See `examples/batch_config.toml` for the config schema and `examples/derived_defs.py` for defining derived variables.

- **`input.py`** parses Athena++ parameter files with `section/key` access; **`athena_read.py`** is (mostly vendored) Athena++ reader code for `.hst` etc.

`docs/` contains per-module reference documentation mirroring this structure — update it when changing public APIs.
