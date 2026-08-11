import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

# for -f argument
import yaaps as ya
import yaaps.decorations as yd
import yaaps.plot2D as yp
from yaaps.plot_formatter import PlotFormatter

ap = argparse.ArgumentParser("Create one 2D color plot per simulation as panels of a shared figure")

ap.add_argument("var", type=str, nargs="?", default=None, help="Variable to plot")
ap.add_argument("-o", "--outputpath", type=str, default=None, help="Path to save at")
ap.add_argument("-t", "--time", type=float, default=1e5, help="Time to plot at")
ap.add_argument(
    "-s",
    "--simdir",
    type=str,
    default=["."],
    nargs="+",
    help="Directories to look for athdf files in",
)
ap.add_argument("-r", "--sampling", type=str, default="xy", help="Plane to plot")
ap.add_argument("-c", "--cmap", type=str, default=None, help="Colormap")
ap.add_argument("-n", "--norm", type=str, default=None, help="-n 'log' for logsscale")
ap.add_argument("-b", "--boundary", type=float, default=None, help="Boundary of the plot")
ap.add_argument("-m", "--meshblocks", action="store_true", help="Draw mesh-block boundaries")
ap.add_argument(
    "-f", "--func", default="None", type=str, help="Modify plot with given function. (calls eval)"
)
ap.add_argument("--vmin", type=float, default=None, help="Minimum of the colorscale")
ap.add_argument("--vmax", type=float, default=None, help="Maximum of the colorscale")
ap.add_argument(
    "-p", "--paper-format", action="store_true", help="Use paper-ready and units format for labels"
)

args = ap.parse_args()

sims = [ya.Simulation(s) for s in args.simdir]

if args.var is None:
    varnames = yd.reverse_var_alias
    av_v = sorted(set(vv for vv, *_ in sims[0].scrape.debug_data_keys().keys()))
    print("Available vars:")
    max_len = max(len(vv) for vv in av_v)
    for vv in av_v:
        if varnames.get(vv):
            print(f"  {vv.ljust(max_len)} -> {varnames[vv]}")
        else:
            print(f"  {vv}")
    exit(0)

if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

func = eval(args.func)

formatter = PlotFormatter("paper" if args.paper_format else "raw")
if args.paper_format:
    args.time = formatter.inverse_convert_time(args.time)


def _label_root(path):
    # Normalize away a legacy "combine"/"output-XXXX" wrapper directory, so
    # labels come out the same whether a sim points directly at its
    # top-level directory or (as before restarts were merged transparently)
    # at a combine/output-XXXX subdirectory within it.
    path = path.rstrip("/")
    base = os.path.basename(path)
    if base == "combine" or base.startswith("output-"):
        return os.path.dirname(path)
    return path


label_roots = [_label_root(sim.path) for sim in sims]
if len(label_roots) > 1:
    common_path = os.path.commonpath(label_roots)
else:
    common_path = os.path.dirname(label_roots[0])
sim_labels = {
    sim.path: root.replace(common_path, "").strip("/")
    for sim, root in zip(sims, label_roots, strict=False)
}

n = len(sims)
fig, axs = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True, constrained_layout=True)
axs = np.atleast_1d(axs)

plots = [
    yp.NativeColorPlot(
        sim,
        var=args.var,
        sampling=args.sampling,
        func=func,
        draw_meshblocks=args.meshblocks,
        ax=ax,
        cbar=False,
        formatter=formatter,
    )
    for sim, ax in zip(sims, axs, strict=False)
]

# Probe all sims at the target time to build one norm shared by every panel
# (a string norm would autoscale each panel independently).
probed = []
actual_times = []
for plot in plots:
    _, data, actual_time = plot.data.load_data(args.time + plot.t_off)
    if func is not None:
        data = func(data)
    probed.append(formatter.convert_data(plot.data.var, data).ravel())
    actual_times.append(actual_time)

color_kwargs = {
    k: v
    for k, v in (("norm", args.norm), ("cmap", args.cmap), ("vmin", args.vmin), ("vmax", args.vmax))
    if v is not None
}
color_kwargs = yd.update_color_kwargs(plots[0].data.var, color_kwargs, np.concatenate(probed))

for plot, actual_time, ax, sim in zip(plots, actual_times, axs, sims, strict=False):
    plot.kwargs.update(color_kwargs)
    plot.plot(args.time)
    t_show = actual_time - plot.t_off
    if args.paper_format:
        scale, unit = formatter.unit_converter.get_conversion("time")
        title_time = f"$t = {t_show * scale:.1f}$ {unit.strip()}"
    else:
        title_time = f"t = {t_show:.1f}"
    ax.set_title(f"{sim_labels[sim.path]} @ {title_time}")
for ax in axs[1:]:
    ax.set_ylabel("")

fig.colorbar(
    plots[-1].ims[-1],
    ax=list(axs),
    label=formatter.format_colorbar_label(plots[0].data.var),
)

if args.boundary is not None:
    for ax in axs:
        ax.set_xlim(-args.boundary, args.boundary)
        ax.set_ylim(-args.boundary, args.boundary)

plt.savefig(args.outputpath, dpi=200, bbox_inches="tight")
print(args.outputpath)
