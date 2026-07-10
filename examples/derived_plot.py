import os
import sys
import tempfile
import argparse
import matplotlib.pyplot as plt
import yaaps as ya
import yaaps.decorations as yd
from derived_defs import vars

# for -f argument
import numpy as np

ap = argparse.ArgumentParser("Create a 2D grid plot using yaaps and save it as png")

ap.add_argument('var', type=str, nargs='?', default=None,
                help="Variable to plot")
ap.add_argument('-o','--outputpath', type=str, default=None,
                help="Path to save at")
ap.add_argument('-t','--time', type=float, default=1e5,
                help="Time to plot at")
ap.add_argument('-s','--simdir', type=str, default='active',
                help="Directory to look for athdf files in")
ap.add_argument('-r','--sampling', type=str, default='xy',
                help="Plane to plot")
ap.add_argument('-c','--cmap', type=str, default=None,
                help="Colormap")
ap.add_argument('-n','--norm', type=str, default=None,
                help="-n 'log' for logsscale")
ap.add_argument('-b', '--boundary', type=float, default=None,
                help="Boundary of the plot")
ap.add_argument('-m','--meshblocks', action="store_true",
                help="Draw mesh-block boundaries")
ap.add_argument('-f','--func', default="None", type=str,
                help="Modify plot with given function. (calls eval)")
ap.add_argument('--vmin', type=float, default=None,
                help="Minimum of the colorscale")
ap.add_argument('--vmax', type=float, default=None,
                help="Maximum of the colorscale")
ap.add_argument('-p', '--paper-format', action='store_true',
                help="Use paper-ready and units format for labels")

args = ap.parse_args()

sim = ya.Simulation(args.simdir)

if args.var is None or args.var not in vars:
    print("Available vars:")
    print(vars.keys())

if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

func = eval(args.func)

if args.paper_format:
    formatter = ya.plot_formatter.PlotFormatter("paper")
    args.time = formatter.inverse_convert_time(args.time)


fig, ax = plt.subplots(1, figsize=(4,3.5))
kwargs = dict(
    norm=args.norm,
    func=func,
    sampling=args.sampling,
    cmap=args.cmap,
    vmin=args.vmin,
    vmax=args.vmax,
    draw_meshblocks=args.meshblocks,
    ax=ax,
    formatter="paper" if args.paper_format else "raw",
)

for k, v in list(kwargs.items()):
    if v is None:
        kwargs.pop(k)

plot = vars[args.var](sim, **kwargs)
plot.plot(time=args.time)

if args.boundary is not None:
    plt.xlim(-args.boundary, args.boundary)
    plt.ylim(-args.boundary, args.boundary)

plt.savefig(args.outputpath, dpi=200, bbox_inches='tight')
print(args.outputpath)
