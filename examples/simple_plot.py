import os
import sys
import tempfile
import argparse
import matplotlib.pyplot as plt
import yaaps as ya

# for -f argument
import numpy as np

ap = argparse.ArgumentParser("Create a 2D grid plot using yaaps and save it as png")

ap.add_argument('var', type=str, help="Variable to plot")
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

if len(sys.argv) == 1:
    sim = ya.Simulation("active")
    av_v = sorted(list(set(vv for vv, *_ in sim.scrape.debug_data_keys().keys())))
    print("Available vars:")
    for vv in av_v:
        print(f"  {vv}")
    exit(0)

args = ap.parse_args()

sim = ya.Simulation(args.simdir)

if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

func = eval(args.func)

fig, ax = plt.subplots(1, figsize=(4,3.5))
kwargs = dict(
    var=args.var,
    time=args.time,
    norm=args.norm,
    func=func,
    sampling=args.sampling,
    cmap=args.cmap,
    draw_meshblocks=args.meshblocks,
    vmin=args.vmin,
    vmax=args.vmax,
    ax=ax,
    paper_format="paper" if args.paper_format else "raw",
)

for k, v in list(kwargs.items()):
    if v is None:
        kwargs.pop(k)

sim.plot2d(**kwargs)

if args.boundary is not None:
    plt.xlim(-args.boundary, args.boundary)
    plt.ylim(-args.boundary, args.boundary)

plt.savefig(args.outputpath, dpi=200, bbox_inches='tight')
print(args.outputpath)
