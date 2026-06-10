"""
Batch plotting script for yaaps.

Reads a TOML configuration file describing multiple plots/animations to produce
and generates them for a list of simulations.  Supports:

- History (hst) plots: all simulations combined in a single figure.
- Simulation-combined plots: a grid of subplots (one per simulation) for a
  single variable at specified times.
- Variable-combined plots: a single figure combining multiple variables for one
  simulation at specified times.
- Animation variants of the above two using parallel frame rendering.
- Optional NativeContourPlot overlays on any 2D plot/animation.

Usage
-----
    python -m yaaps.batch_plots \\
        --sims /path/to/sim1 /path/to/sim2 \\
        --output-dir ./plots \\
        --config plots.toml \\
        --cpus 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
from math import isqrt
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import yaaps as ya  # noqa: E402
import yaaps.plot2D as yp  # noqa: E402
import yaaps.decorations as yd  # noqa: E402
from yaaps.datatypes import Native  # noqa: E402
from yaaps.plot_formatter import PlotFormatter  # noqa: E402


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _split(N: int) -> tuple[int, int]:
    """Return (rows, cols) for a grid with N panels."""
    for n in range(isqrt(N), 0, -1):
        if not (N % n):
            return n, N // n
    raise ValueError(f"Cannot split {N}")


def _eval_func(func_str: str | None) -> Callable | None:
    """Evaluate a function string or return None."""
    if func_str is None or func_str in ("None", "none", ""):
        return None
    if func_str in ("id", "identity"):
        return None
    if func_str in ("relabs", "absrel"):
        return lambda d: np.abs(d / d[0] - 1)
    if func_str == "absdiff":
        return lambda d: np.abs(d - d[0])
    if func_str == "diff":
        return lambda d: d - d[0]
    if func_str == "inv":
        return np.reciprocal
    if func_str == "abs":
        return np.abs
    result = eval(func_str)  # noqa: S307
    if isinstance(result, (int, float)):
        factor = result
        return lambda d: d * factor
    if callable(result):
        return result
    raise ValueError(f"'{func_str}' does not evaluate to a callable")


def _build_contour_plots(
    sim: ya.Simulation,
    ax: plt.Axes,
    contour_cfg: dict[str, Any],
    formatter: str,
    sampling: str,
) -> list[yp.NativeContourPlot]:
    """Build NativeContourPlot objects from a contour config dict."""
    plots = []
    # contour_cfg can be a single dict or a list of dicts
    items = contour_cfg if isinstance(contour_cfg, list) else [contour_cfg]
    for item in items:
        kwargs: dict[str, Any] = {}
        kwargs["var"] = item["var"]
        kwargs["sampling"] = item.get("sampling", sampling)
        kwargs["bounds"] = item.get("bounds", 50.0)
        kwargs["N_points"] = item.get("N_points", 200)
        kwargs["ax"] = ax
        kwargs["formatter"] = formatter
        kwargs["cbar"] = item.get("cbar", False)
        # Contour-specific matplotlib kwargs
        if "levels" in item:
            kwargs["levels"] = item["levels"]
        if "colors" in item:
            kwargs["colors"] = item["colors"]
        if "linewidths" in item:
            kwargs["linewidths"] = item["linewidths"]
        if "linestyles" in item:
            kwargs["linestyles"] = item["linestyles"]
        if "cmap" in item:
            kwargs["cmap"] = item["cmap"]
        if "norm" in item:
            kwargs["norm"] = item["norm"]
        if "vmin" in item:
            kwargs["vmin"] = item["vmin"]
        if "vmax" in item:
            kwargs["vmax"] = item["vmax"]
        if "alpha" in item:
            kwargs["alpha"] = item["alpha"]
        plots.append(yp.NativeContourPlot(sim=sim, **kwargs))
    return plots


def _get_times_for_sim(
    sim: ya.Simulation,
    var: str,
    sampling: str,
    time_list: list[float] | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
    time_every: int = 1,
) -> np.ndarray:
    """Get available times for a simulation variable, filtered."""
    data = Native(sim, var, sampling)
    times = data.time_range
    if time_min is not None:
        times = times[times >= time_min]
    if time_max is not None:
        times = times[times <= time_max]
    times = times[::time_every]
    if time_list is not None:
        # Find nearest available time for each requested time
        selected = []
        for t in time_list:
            idx = np.argmin(np.abs(times - t))
            selected.append(times[idx])
        times = np.unique(selected)
    return times


# ---------------------------------------------------------------------------
# Plot type: hst (history)
# ---------------------------------------------------------------------------

def _plot_hst(
    sims: list[ya.Simulation],
    output_dir: str,
    section_name: str,
    cfg: dict[str, Any],
) -> None:
    """Generate hst plots combining all simulations."""
    variables = cfg.get("vars", ["max_rho"])
    if isinstance(variables, str):
        variables = [variables]
    colors = cfg.get("colors", [f"C{i}" for i in range(len(sims))])
    xvar = cfg.get("xvar", "time")
    xlim = cfg.get("xlim", None)
    xlog = cfg.get("xlog", False)
    ylog_vars = cfg.get("ylog", [])
    ylim_dict = cfg.get("ylim", {})
    no_auto_log = cfg.get("no_auto_log", False)
    funcs_cfg = cfg.get("funcs", {})
    dpi = cfg.get("dpi", 200)
    figsize = cfg.get("figsize", None)

    auto_log_keys = [
        "mass",
        "max_sc_nG_00", "max_sc_nG_01", "max_sc_nG_02",
        "max_sc_E_00", "max_sc_E_01", "max_sc_E_02",
        "max_sc_n_00", "max_sc_n_01", "max_sc_n_02",
        "max_sc_J_00", "max_sc_J_01", "max_sc_J_02",
    ]

    # Parse vars (support comma-separated for multi-line subplots)
    parsed_vars: list[str | list[str]] = []
    for v in variables:
        if "," in v:
            parsed_vars.append([x.strip() for x in v.split(",")])
        else:
            parsed_vars.append(v)

    # Parse funcs
    funcs: dict[str, Callable] = {}
    func_names: dict[str, str] = {}
    for var_key, f_str in funcs_cfg.items():
        funcs[var_key] = _eval_func(f_str)  # type: ignore[assignment]
        func_names[var_key] = f_str

    m, n = _split(len(parsed_vars))
    if figsize is None:
        figsize = (n * 7, m * 4)
    fig, axs = plt.subplots(m, n, figsize=figsize, sharex=True)
    axs = np.atleast_1d(axs)

    if len(sims) > 1:
        common_path = os.path.commonpath([sim.path for sim in sims])
    else:
        common_path = os.path.dirname(os.path.dirname(sims[0].path))

    suffix = ""
    bases = [os.path.basename(sim.path) for sim in sims]
    if all(b == bases[0] for b in bases):
        suffix = "/" + bases[0]

    for var, ax in zip(parsed_vars, axs.flat):
        for sim, c in zip(sims, colors):
            name = sim.path.replace(common_path, "").strip("/").replace(suffix, "")
            if isinstance(var, list):
                for v, ls in zip(var, ("-", "--", ":", "-.")):
                    label = v if sim is sims[0] else None
                    src = sim.hst
                    data = src[v]
                    if v in funcs and funcs[v] is not None:
                        data = funcs[v](data)
                    ax.plot(src[xvar], data, c=c, ls=ls, label=label)
            else:
                src = sim.hst
                data = src[var]
                if var in funcs and funcs[var] is not None:
                    data = funcs[var](data)
                ax.plot(src[xvar], data, c=c, label=name)
        ax.set_xlabel(xvar)

        if isinstance(var, list):
            ylabel = " ".join(var)
            ax.set_ylabel(ylabel)
            for v in var:
                if v in ylog_vars or (not no_auto_log and v in auto_log_keys):
                    ax.set_yscale("log")
                    break
        else:
            ylabel = var
            if var in func_names:
                ylabel = f"{func_names[var]}({var})"
            ax.set_ylabel(ylabel)
            if var in ylog_vars or (not no_auto_log and var in auto_log_keys):
                ax.set_yscale("log")

    # Legends and limits
    sim_legend_exists = False
    for var, ax in zip(parsed_vars, axs.flat):
        if isinstance(var, list):
            ax.legend()
        elif not sim_legend_exists:
            ax.legend()
            sim_legend_exists = True
        # ylim
        if isinstance(var, list):
            for v in var:
                if v in ylim_dict:
                    lims = ylim_dict[v]
                    ax.set_ylim(lims[0], lims[1])
        elif var in ylim_dict:
            lims = ylim_dict[var]
            ax.set_ylim(lims[0], lims[1])

    if xlog:
        for ax in axs.flat:
            ax.set_xscale("log")
    if xlim is not None:
        for ax in axs.flat:
            ax.set_xlim(xlim[0], xlim[1])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{section_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot type: sim_combined (grid of sims, one per subplot)
# ---------------------------------------------------------------------------

def _render_sim_combined_frame(args_tuple: tuple) -> str:
    """Render a single sim-combined frame (for use in pool or serial)."""
    (sims_paths, var, time, sampling, boundary, cfg, output_dir, name_prefix) = args_tuple

    func = _eval_func(cfg.get("func"))
    formatter_mode = "paper" if cfg.get("paper_format", False) else "raw"
    dpi = cfg.get("dpi", 200)
    figsize = cfg.get("figsize")
    cmap = cfg.get("cmap")
    norm = cfg.get("norm")
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    meshblocks = cfg.get("meshblocks", False)
    contour_cfg = cfg.get("contour")

    n_sims = len(sims_paths)
    rows, cols = _split(n_sims)
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, sim_path in enumerate(sims_paths):
        sim = ya.Simulation(sim_path)
        ax = axs.flat[idx]

        plot_kwargs: dict[str, Any] = dict(
            var=var,
            sampling=sampling,
            func=func,
            draw_meshblocks=meshblocks,
            ax=ax,
            formatter=formatter_mode,
        )
        if cmap is not None:
            plot_kwargs["cmap"] = cmap
        if norm is not None:
            plot_kwargs["norm"] = norm
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

        color_plot = yp.NativeColorPlot(sim=sim, **plot_kwargs)
        color_plot.plot(time)

        # Contour overlay
        if contour_cfg is not None:
            contour_plots = _build_contour_plots(
                sim, ax, contour_cfg, formatter_mode, sampling
            )
            for cp in contour_plots:
                cp.plot(time)

        if boundary is not None:
            ax.set_xlim(-boundary, boundary)
            ax.set_ylim(-boundary, boundary)

        ax.set_title(f"{sim.name}\n{ax.get_title()}")

    # Hide unused axes
    for idx in range(n_sims, rows * cols):
        axs.flat[idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{name_prefix}.pdf")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    # Also save PNG
    fname_png = os.path.join(output_dir, f"{name_prefix}.png")
    fig.savefig(fname_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_sim_combined(
    sims: list[ya.Simulation],
    output_dir: str,
    section_name: str,
    cfg: dict[str, Any],
    n_cpus: int = 1,
) -> None:
    """Generate sim-combined plots (grid of subplots, one per sim)."""
    keys = cfg.get("vars", cfg.get("keys", ["hydro.prim.rho"]))
    if isinstance(keys, str):
        keys = [keys]
    sampling = cfg.get("sampling", "xy")
    boundary = cfg.get("boundary")
    time_list = cfg.get("times")
    time_min = cfg.get("time_min")
    time_max = cfg.get("time_max")
    time_every = cfg.get("time_every", 1)

    # Get times from first sim/first var
    times = _get_times_for_sim(
        sims[0], keys[0], sampling, time_list, time_min, time_max, time_every
    )

    sim_paths = [sim.path for sim in sims]
    tasks = []
    for var in keys:
        var_dir = os.path.join(output_dir, section_name)
        os.makedirs(var_dir, exist_ok=True)
        for ti, time in enumerate(times):
            name_prefix = f"{var}_{ti:04d}_t{time:.2f}"
            tasks.append(
                (sim_paths, var, time, sampling, boundary, cfg, var_dir, name_prefix)
            )

    if n_cpus > 1 and len(tasks) > 1:
        with multiprocessing.Pool(n_cpus) as pool:
            results = list(pool.imap_unordered(_render_sim_combined_frame, tasks))
    else:
        results = [_render_sim_combined_frame(t) for t in tasks]

    print(f"  Saved {len(results)} plots to {os.path.join(output_dir, section_name)}/")


# ---------------------------------------------------------------------------
# Plot type: var_combined (multiple vars in one figure per time/sim)
# ---------------------------------------------------------------------------

def _render_var_combined_frame(args_tuple: tuple) -> str:
    """Render a single var-combined frame."""
    (sim_path, variables, time, sampling, boundary, cfg, output_dir, name_prefix) = args_tuple

    formatter_mode = "paper" if cfg.get("paper_format", False) else "raw"
    dpi = cfg.get("dpi", 200)
    figsize = cfg.get("figsize")
    meshblocks = cfg.get("meshblocks", False)
    contour_cfg = cfg.get("contour")

    n_vars = len(variables)
    rows, cols = _split(n_vars)
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    sim = ya.Simulation(sim_path)

    for idx, var_cfg in enumerate(variables):
        ax = axs.flat[idx]

        if isinstance(var_cfg, str):
            var_name = var_cfg
            var_opts: dict[str, Any] = {}
        else:
            var_name = var_cfg["var"]
            var_opts = {k: v for k, v in var_cfg.items() if k != "var"}

        func = _eval_func(var_opts.get("func", cfg.get("func")))
        cmap = var_opts.get("cmap", cfg.get("cmap"))
        norm = var_opts.get("norm", cfg.get("norm"))
        vmin = var_opts.get("vmin", cfg.get("vmin"))
        vmax = var_opts.get("vmax", cfg.get("vmax"))

        plot_kwargs: dict[str, Any] = dict(
            var=var_name,
            sampling=var_opts.get("sampling", sampling),
            func=func,
            draw_meshblocks=meshblocks,
            ax=ax,
            formatter=formatter_mode,
        )
        if cmap is not None:
            plot_kwargs["cmap"] = cmap
        if norm is not None:
            plot_kwargs["norm"] = norm
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

        color_plot = yp.NativeColorPlot(sim=sim, **plot_kwargs)
        color_plot.plot(time)

        # Per-variable contour
        var_contour = var_opts.get("contour", contour_cfg)
        if var_contour is not None:
            contour_plots = _build_contour_plots(
                sim, ax, var_contour, formatter_mode,
                var_opts.get("sampling", sampling),
            )
            for cp in contour_plots:
                cp.plot(time)

        if boundary is not None:
            ax.set_xlim(-boundary, boundary)
            ax.set_ylim(-boundary, boundary)

    # Hide unused axes
    for idx in range(n_vars, rows * cols):
        axs.flat[idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{name_prefix}.pdf")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    fname_png = os.path.join(output_dir, f"{name_prefix}.png")
    fig.savefig(fname_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_var_combined(
    sims: list[ya.Simulation],
    output_dir: str,
    section_name: str,
    cfg: dict[str, Any],
    n_cpus: int = 1,
) -> None:
    """Generate var-combined plots (multiple vars per figure, per sim/time)."""
    variables = cfg["vars"]  # list of var names or dicts with per-var options
    sampling = cfg.get("sampling", "xy")
    boundary = cfg.get("boundary")
    time_list = cfg.get("times")
    time_min = cfg.get("time_min")
    time_max = cfg.get("time_max")
    time_every = cfg.get("time_every", 1)

    # Determine first var name for time range
    first_var = variables[0] if isinstance(variables[0], str) else variables[0]["var"]

    tasks = []
    for sim in sims:
        times = _get_times_for_sim(
            sim, first_var, sampling, time_list, time_min, time_max, time_every
        )
        sim_dir = os.path.join(output_dir, section_name, sim.name)
        os.makedirs(sim_dir, exist_ok=True)
        for ti, time in enumerate(times):
            name_prefix = f"vars_{ti:04d}_t{time:.2f}"
            tasks.append(
                (sim.path, variables, time, sampling, boundary, cfg, sim_dir, name_prefix)
            )

    if n_cpus > 1 and len(tasks) > 1:
        with multiprocessing.Pool(n_cpus) as pool:
            results = list(pool.imap_unordered(_render_var_combined_frame, tasks))
    else:
        results = [_render_var_combined_frame(t) for t in tasks]

    print(f"  Saved {len(results)} plots to {os.path.join(output_dir, section_name)}/")


# ---------------------------------------------------------------------------
# Animation type: sim_combined_anim (parallel frame rendering)
# ---------------------------------------------------------------------------

# Worker state for sim_combined animation
_anim_sim_sims: list[ya.Simulation] | None = None
_anim_sim_cfg: dict[str, Any] | None = None


def _sim_anim_worker_init(sim_paths: list[str], cfg: dict[str, Any]) -> None:
    """Worker initializer for sim-combined animation."""
    global _anim_sim_sims, _anim_sim_cfg
    matplotlib.use("Agg")
    _anim_sim_sims = [ya.Simulation(p) for p in sim_paths]
    _anim_sim_cfg = cfg


def _sim_anim_render_frame(task: tuple[int, float]) -> str:
    """Render one frame of a sim-combined animation."""
    frame_idx, time = task
    cfg = _anim_sim_cfg
    sims = _anim_sim_sims

    var = cfg["var"]
    sampling = cfg.get("sampling", "xy")
    boundary = cfg.get("boundary")
    func = _eval_func(cfg.get("func"))
    formatter_mode = "paper" if cfg.get("paper_format", False) else "raw"
    dpi = cfg.get("dpi", 200)
    figsize = cfg.get("figsize")
    cmap = cfg.get("cmap")
    norm = cfg.get("norm")
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    meshblocks = cfg.get("meshblocks", False)
    contour_cfg = cfg.get("contour")

    n_sims = len(sims)
    rows, cols = _split(n_sims)
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, sim in enumerate(sims):
        ax = axs.flat[idx]
        plot_kwargs: dict[str, Any] = dict(
            var=var,
            sampling=sampling,
            func=func,
            draw_meshblocks=meshblocks,
            ax=ax,
            formatter=formatter_mode,
        )
        if cmap is not None:
            plot_kwargs["cmap"] = cmap
        if norm is not None:
            plot_kwargs["norm"] = norm
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

        color_plot = yp.NativeColorPlot(sim=sim, **plot_kwargs)
        color_plot.plot(time)

        if contour_cfg is not None:
            contour_plots = _build_contour_plots(
                sim, ax, contour_cfg, formatter_mode, sampling
            )
            for cp in contour_plots:
                cp.plot(time)

        if boundary is not None:
            ax.set_xlim(-boundary, boundary)
            ax.set_ylim(-boundary, boundary)

        ax.set_title(f"{sim.name}\n{ax.get_title()}")

    for idx in range(n_sims, rows * cols):
        axs.flat[idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(cfg["output_dir"], f"frame_{frame_idx:06d}.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


def _anim_sim_combined(
    sims: list[ya.Simulation],
    output_dir: str,
    section_name: str,
    cfg: dict[str, Any],
    n_cpus: int = 1,
) -> None:
    """Generate sim-combined animation with parallel rendering."""
    var = cfg.get("var", cfg.get("vars", "hydro.prim.rho"))
    if isinstance(var, list):
        var = var[0]
    cfg["var"] = var
    sampling = cfg.get("sampling", "xy")
    time_min = cfg.get("time_min")
    time_max = cfg.get("time_max")
    time_every = cfg.get("time_every", 1)
    fps = cfg.get("fps", "24")

    # Get union of all sim times
    all_times = []
    for sim in sims:
        data = Native(sim, var, sampling)
        all_times.append(data.time_range)
    times = np.unique(np.concatenate(all_times))

    if time_min is not None:
        times = times[times >= time_min]
    if time_max is not None:
        times = times[times <= time_max]
    times = times[::time_every]

    if len(times) == 0:
        print(f"  WARNING: No frames for {section_name}, skipping.")
        return

    frames_dir = os.path.join(output_dir, section_name, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    cfg["output_dir"] = frames_dir

    sim_paths = [sim.path for sim in sims]
    tasks = list(enumerate(times))
    n_workers = max(1, n_cpus)

    print(f"  Rendering {len(tasks)} frames with {n_workers} workers...")

    if n_workers > 1:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_sim_anim_worker_init,
            initargs=(sim_paths, cfg),
        ) as pool:
            results = list(pool.imap_unordered(_sim_anim_render_frame, tasks))
    else:
        _sim_anim_worker_init(sim_paths, cfg)
        results = [_sim_anim_render_frame(t) for t in tasks]

    # Create MP4
    output_mp4 = os.path.join(output_dir, section_name, f"{section_name}.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%06d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-movflags", "+faststart",
                output_mp4,
            ],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        print(f"  Created: {output_mp4}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  WARNING: ffmpeg failed; frames saved in {frames_dir}")

    print(f"  Saved {len(results)} frames to {frames_dir}/")


# ---------------------------------------------------------------------------
# Animation type: var_combined_anim (parallel frame rendering)
# ---------------------------------------------------------------------------

_anim_var_sim: ya.Simulation | None = None
_anim_var_cfg: dict[str, Any] | None = None


def _var_anim_worker_init(sim_path: str, cfg: dict[str, Any]) -> None:
    """Worker initializer for var-combined animation."""
    global _anim_var_sim, _anim_var_cfg
    matplotlib.use("Agg")
    _anim_var_sim = ya.Simulation(sim_path)
    _anim_var_cfg = cfg


def _var_anim_render_frame(task: tuple[int, float]) -> str:
    """Render one frame of a var-combined animation."""
    frame_idx, time = task
    cfg = _anim_var_cfg
    sim = _anim_var_sim

    variables = cfg["vars"]
    sampling = cfg.get("sampling", "xy")
    boundary = cfg.get("boundary")
    formatter_mode = "paper" if cfg.get("paper_format", False) else "raw"
    dpi = cfg.get("dpi", 200)
    figsize = cfg.get("figsize")
    meshblocks = cfg.get("meshblocks", False)
    contour_cfg = cfg.get("contour")

    n_vars = len(variables)
    rows, cols = _split(n_vars)
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, var_cfg in enumerate(variables):
        ax = axs.flat[idx]

        if isinstance(var_cfg, str):
            var_name = var_cfg
            var_opts: dict[str, Any] = {}
        else:
            var_name = var_cfg["var"]
            var_opts = {k: v for k, v in var_cfg.items() if k != "var"}

        func = _eval_func(var_opts.get("func", cfg.get("func")))
        cmap = var_opts.get("cmap", cfg.get("cmap"))
        norm_val = var_opts.get("norm", cfg.get("norm"))
        vmin = var_opts.get("vmin", cfg.get("vmin"))
        vmax = var_opts.get("vmax", cfg.get("vmax"))

        plot_kwargs: dict[str, Any] = dict(
            var=var_name,
            sampling=var_opts.get("sampling", sampling),
            func=func,
            draw_meshblocks=meshblocks,
            ax=ax,
            formatter=formatter_mode,
        )
        if cmap is not None:
            plot_kwargs["cmap"] = cmap
        if norm_val is not None:
            plot_kwargs["norm"] = norm_val
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

        color_plot = yp.NativeColorPlot(sim=sim, **plot_kwargs)
        color_plot.plot(time)

        var_contour = var_opts.get("contour", contour_cfg)
        if var_contour is not None:
            contour_plots = _build_contour_plots(
                sim, ax, var_contour, formatter_mode,
                var_opts.get("sampling", sampling),
            )
            for cp in contour_plots:
                cp.plot(time)

        if boundary is not None:
            ax.set_xlim(-boundary, boundary)
            ax.set_ylim(-boundary, boundary)

    for idx in range(n_vars, rows * cols):
        axs.flat[idx].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(cfg["output_dir"], f"frame_{frame_idx:06d}.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


def _anim_var_combined(
    sims: list[ya.Simulation],
    output_dir: str,
    section_name: str,
    cfg: dict[str, Any],
    n_cpus: int = 1,
) -> None:
    """Generate var-combined animation with parallel rendering per sim."""
    variables = cfg["vars"]
    sampling = cfg.get("sampling", "xy")
    time_min = cfg.get("time_min")
    time_max = cfg.get("time_max")
    time_every = cfg.get("time_every", 1)
    fps = cfg.get("fps", "24")

    first_var = variables[0] if isinstance(variables[0], str) else variables[0]["var"]

    for sim in sims:
        data = Native(sim, first_var, sampling)
        times = data.time_range
        if time_min is not None:
            times = times[times >= time_min]
        if time_max is not None:
            times = times[times <= time_max]
        times = times[::time_every]

        if len(times) == 0:
            print(f"  WARNING: No frames for {section_name}/{sim.name}, skipping.")
            continue

        frames_dir = os.path.join(output_dir, section_name, sim.name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        cfg_copy = dict(cfg)
        cfg_copy["output_dir"] = frames_dir

        tasks = list(enumerate(times))
        n_workers = max(1, n_cpus)

        print(f"  [{sim.name}] Rendering {len(tasks)} frames with {n_workers} workers...")

        if n_workers > 1:
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_var_anim_worker_init,
                initargs=(sim.path, cfg_copy),
            ) as pool:
                results = list(pool.imap_unordered(_var_anim_render_frame, tasks))
        else:
            _var_anim_worker_init(sim.path, cfg_copy)
            results = [_var_anim_render_frame(t) for t in tasks]

        # Create MP4
        output_mp4 = os.path.join(
            output_dir, section_name, sim.name, f"{section_name}.mp4"
        )
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "frame_%06d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-movflags", "+faststart",
                    output_mp4,
                ],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            print(f"  Created: {output_mp4}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  WARNING: ffmpeg failed; frames saved in {frames_dir}")

        print(f"  [{sim.name}] Saved {len(results)} frames.")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PLOT_TYPES = {
    "hst": _plot_hst,
    "sim_combined": _plot_sim_combined,
    "var_combined": _plot_var_combined,
    "sim_combined_anim": _anim_sim_combined,
    "var_combined_anim": _anim_var_combined,
}


def main() -> None:
    """Entry point for batch plot generation."""
    ap = argparse.ArgumentParser(
        description="Batch plotting script: generate multiple plots/animations "
        "from a TOML configuration file."
    )
    ap.add_argument(
        "--sims", "-s",
        type=str, nargs="+", required=True,
        help="Paths to simulation directories",
    )
    ap.add_argument(
        "--output-dir", "-o",
        type=str, default="./batch_plots",
        help="Directory to save all generated plots/animations",
    )
    ap.add_argument(
        "--config", "-c",
        type=str, required=True,
        help="Path to TOML configuration file describing plots to generate",
    )
    ap.add_argument(
        "--cpus", "-n",
        type=int, default=1,
        help="Number of CPUs for parallelization",
    )

    args = ap.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Load simulations
    sims: list[ya.Simulation] = []
    for sim_path in args.sims:
        try:
            sims.append(ya.Simulation(sim_path))
        except FileNotFoundError:
            print(f"WARNING: No parfile in {sim_path}, skipping", file=sys.stderr)

    if not sims:
        print("ERROR: No valid simulations found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each section
    for section_name, section_cfg in config.items():
        plot_type = section_cfg.get("type")
        if plot_type is None:
            print(
                f"WARNING: Section [{section_name}] has no 'type' key, skipping.",
                file=sys.stderr,
            )
            continue

        if plot_type not in PLOT_TYPES:
            print(
                f"WARNING: Unknown plot type '{plot_type}' in [{section_name}], skipping.",
                file=sys.stderr,
            )
            continue

        print(f"Processing [{section_name}] (type={plot_type})...")

        handler = PLOT_TYPES[plot_type]
        if plot_type == "hst":
            handler(sims, args.output_dir, section_name, section_cfg)
        else:
            handler(sims, args.output_dir, section_name, section_cfg, args.cpus)


if __name__ == "__main__":
    main()
