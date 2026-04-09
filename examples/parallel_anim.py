"""
Parallel animation example for yaaps.

Renders a range of simulation snapshots to individual PNG files using
multiple worker processes.  Workers are initialized once (loading the
simulation from disk a single time per process) and then receive
individual frame tasks via Pool.imap_unordered, which provides dynamic
load-balancing across workers.

Why this approach?
------------------
- The serial ``simple_anim.py`` reuses one figure/axes object across all
  frames.  That design is efficient in a single process but cannot be
  shared safely across processes.
- By creating a **fresh** figure/axes/plot per task and closing it when
  done we avoid any shared matplotlib state between workers.
- ``Pool.imap_unordered`` distributes tasks dynamically: a fast worker
  immediately picks up the next pending frame instead of waiting for
  slower peers (better load balancing than ``Pool.map``).
- Loading the simulation object in the worker *initializer* (once per
  process, not once per frame) amortizes I/O start-up cost.

Usage
-----
  python parallel_anim.py rho -s /path/to/sim -o frames/ -w 8

Assemble frames into a video afterwards, e.g. with ffmpeg::

  ffmpeg -framerate 24 -i frames/frame_%06d.png -c:v libx264 out.mp4
"""

import os
import argparse
import multiprocessing

# Set the non-interactive Agg backend *before* any other matplotlib import.
# This is required for headless (no-display) rendering and must happen at
# module level so that it is executed in every worker process as well.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 (import after matplotlib.use)

import numpy as np  # noqa: E402  (available in worker func-string eval scope)
from tqdm import tqdm  # noqa: E402

import yaaps as ya  # noqa: E402
import yaaps.plot2D as yp  # noqa: E402
from yaaps.datatypes import Native  # noqa: E402
from yaaps.plot_formatter import PlotFormatter  # noqa: E402


# ---------------------------------------------------------------------------
# Per-worker global state – set once by _worker_init, reused for every frame.
# ---------------------------------------------------------------------------
_sim = None        # yaaps.Simulation loaded for this worker
_worker_cfg = None  # dict with all rendering options


def _worker_init(sim_path: str, cfg: dict) -> None:
    """Initialize a worker process.

    Called exactly once per worker when the Pool starts (not once per
    frame task), so the simulation is loaded from disk only once per
    process.  The loaded objects are stored in module-level globals so
    that ``_render_frame`` can access them without passing them through
    the task queue.

    Args:
        sim_path: Path to the simulation output directory.
        cfg: Dict of rendering options shared by all frames in this run.
    """
    global _sim, _worker_cfg
    # Explicitly set the backend in each spawned worker (important on
    # macOS/Windows where 'spawn' is the default start method and the
    # parent's matplotlib state may not be inherited).
    matplotlib.use('Agg')
    _sim = ya.Simulation(sim_path)
    _worker_cfg = cfg


def _render_frame(task: tuple) -> str:
    """Render a single animation frame to a PNG file.

    Each invocation creates its own figure/axes/plot objects to remain
    completely isolated from other workers and from other frames in the
    same worker.  The figure is closed immediately after saving to
    prevent memory growth over long runs.

    Args:
        task: ``(frame_index, time)`` tuple identifying the frame.

    Returns:
        Path to the saved PNG file.
    """
    frame_idx, time = task
    cfg = _worker_cfg

    # Re-evaluate the transformation function string inside each worker.
    # Passing a callable through the task queue would require pickling it,
    # which fails for lambdas.  Passing the source string and eval-ing it
    # here is the simplest portable solution.
    func = eval(cfg.get('func_str', 'None'))  # noqa: S307

    figsize = cfg.get('figsize') or (6, 4)
    fig, ax = plt.subplots(1, figsize=figsize)

    boundary = cfg.get('boundary')
    if boundary is not None:
        ax.set_xlim(-boundary, boundary)
        ax.set_ylim(-boundary, boundary)

    # Build NativeColorPlot kwargs, mirroring simple_anim.py.
    plot_kwargs = dict(
        var=cfg['var'],
        norm=cfg.get('norm'),
        func=func,
        sampling=cfg.get('sampling', 'xy'),
        cmap=cfg.get('cmap'),
        vmin=cfg.get('vmin'),
        vmax=cfg.get('vmax'),
        draw_meshblocks=cfg.get('draw_meshblocks', False),
        ax=ax,
        formatter='paper' if cfg.get('paper_format') else 'raw',
    )
    # Remove None values so NativeColorPlot can apply its own defaults.
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

    plot = yp.NativeColorPlot(sim=_sim, **plot_kwargs)
    plot.plot(time)

    fname = os.path.join(cfg['output_dir'], f"{cfg['prefix']}{frame_idx:06d}.png")
    fig.savefig(fname, dpi=cfg['dpi'], bbox_inches='tight')
    # Release figure memory immediately; the next frame will create a new one.
    plt.close(fig)

    return fname


# ---------------------------------------------------------------------------
# Entry point – must be guarded by if __name__ == '__main__' so that the
# 'spawn' start method (default on macOS and Windows) does not recursively
# execute the setup code in every worker process.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Render animation frames as PNGs in parallel using multiprocessing"
    )

    ap.add_argument('var', type=str,
                    help="Variable to plot")
    ap.add_argument('-s', '--simdir', type=str, default='active',
                    help="Directory to look for athdf files in")
    ap.add_argument('-o', '--output-dir', type=str, default=None,
                    help="Output directory for PNG frames "
                         "(default: VAR_SAMPLING)")
    ap.add_argument('-w', '--workers', type=int, default=None,
                    help="Number of worker processes "
                         "(default: os.cpu_count())")
    ap.add_argument('-r', '--sampling', type=str, default='xy',
                    help="Plane to plot")
    ap.add_argument('-b', '--boundary', type=float, default=None,
                    help="Boundary of the plot")
    ap.add_argument('--time_min', type=float, default=None,
                    help="Time to start at")
    ap.add_argument('--time_max', type=float, default=None,
                    help="Time to end at")
    ap.add_argument('--time_every', type=int, default=1,
                    help="Create frames at every nth output time")
    ap.add_argument('-m', '--meshblocks', action='store_true',
                    help="Draw mesh-block boundaries")
    ap.add_argument('-f', '--func', default='None', type=str,
                    help="Modify plot with given function (calls eval)")
    ap.add_argument('-c', '--cmap', type=str, default=None,
                    help="Colormap")
    ap.add_argument('-n', '--norm', type=str, default=None,
                    help="-n 'log' for logscale")
    ap.add_argument('--vmin', type=float, default=None,
                    help="Minimum of the colorscale")
    ap.add_argument('--vmax', type=float, default=None,
                    help="Maximum of the colorscale")
    ap.add_argument('-p', '--paper-format', action='store_true',
                    help="Use paper-ready and units format for labels")
    ap.add_argument('--dpi', type=int, default=300,
                    help="DPI for output images (default: 300)")
    ap.add_argument('--figsize', type=float, nargs=2, default=None,
                    metavar=('WIDTH', 'HEIGHT'),
                    help="Figure size in inches (e.g. --figsize 6 4)")

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load the simulation in the main process only to query the available
    # time range.  Workers will load their own copies via _worker_init.
    # ------------------------------------------------------------------
    sim = ya.Simulation(args.simdir)
    data = Native(sim, args.var, args.sampling)
    times = data.time_range

    # When paper-format is requested the user supplies time limits in
    # physical units; convert them back to code units before filtering,
    # matching the behaviour of simple_anim.py.
    if args.paper_format:
        formatter = PlotFormatter('paper')
        if args.time_min is not None:
            args.time_min = formatter.inverse_convert_time(args.time_min)
        if args.time_max is not None:
            args.time_max = formatter.inverse_convert_time(args.time_max)

    if args.time_min is not None:
        times = times[times >= args.time_min]
    if args.time_max is not None:
        times = times[times <= args.time_max]
    times = times[::args.time_every]

    if len(times) == 0:
        print("No frames in the specified time range.")
        raise SystemExit(1)

    output_dir = args.output_dir or f"{args.var}_{args.sampling}"
    os.makedirs(output_dir, exist_ok=True)

    # Pack all rendering options into a dict that the worker initializer
    # stores once per process.  Frame tasks only carry frame index + time,
    # keeping the per-task pickling overhead minimal.
    worker_cfg = dict(
        var=args.var,
        sampling=args.sampling,
        boundary=args.boundary,
        norm=args.norm,
        # Pass the raw string so workers can eval() it; lambdas are not
        # picklable and cannot be sent through the task queue directly.
        func_str=args.func,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        draw_meshblocks=args.meshblocks,
        paper_format=args.paper_format,
        dpi=args.dpi,
        figsize=tuple(args.figsize) if args.figsize else None,
        output_dir=output_dir,
        prefix='frame_',
    )

    n_workers = args.workers if args.workers is not None else os.cpu_count()
    n_workers = max(1, n_workers or 1)

    tasks = list(enumerate(times))
    print(f"Rendering {len(tasks)} frames with {n_workers} workers "
          f"into '{output_dir}'...")

    # imap_unordered distributes tasks dynamically: as soon as a worker
    # finishes one frame it picks up the next pending one, keeping all
    # cores busy even when individual frame render times vary.
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(args.simdir, worker_cfg),
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(_render_frame, tasks),
            total=len(tasks),
            desc='Frames',
            unit='frame',
        ))

    print(f"Done. {len(results)} frames saved to '{output_dir}'.")
