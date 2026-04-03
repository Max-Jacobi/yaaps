import argparse
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import subprocess

import yaaps as ya
import yaaps.plot2D as yp
import yaaps.decorations as yd

ap = argparse.ArgumentParser("Make an animation and save the frames as png")

ap.add_argument('var', type=str, nargs='?', default=None,
                help="Variable to plot")
ap.add_argument('-o','--outputpath', type=str, default=None,
                help="Path to save at")
ap.add_argument('-s','--simdir', type=str, default='active',
                help="Directory to look for athdf files in")
ap.add_argument('-r','--sampling', type=str, default='xy',
                help="Plane to plot")
ap.add_argument('-b', '--boundary', type=float, default=None,
                help="Boundary of the plot")
ap.add_argument('--time_min', type=float, default=None,
                help="Time to start at")
ap.add_argument('--time_max', type=float, default=None,
                help="Time to end at")
ap.add_argument('--time_every', type=int, default=1,
                help="Create frames at every nth output time")
ap.add_argument('-m','--meshblocks', action="store_true",
                help="Draw mesh-block boundaries")
ap.add_argument('-f','--func', default="None", type=str,
                help="Modify plot with given function. (calls eval)")
ap.add_argument('-c','--cmap', type=str, default=None,
                help="Colormap")
ap.add_argument('-n','--norm', type=str, default=None,
                help="-n 'log' for logsscale")
ap.add_argument('--vmin', type=float, default=None,
                help="Minimum of the colorscale")
ap.add_argument('--vmax', type=float, default=None,
                help="Maximum of the colorscale")
ap.add_argument('-p', '--paper-format', action='store_true',
                help="Use paper-ready and units format for labels")
ap.add_argument('--fps', type=str, default="2",
                help="Number of fps for the mp4 output")

args = ap.parse_args()

sim = ya.Simulation(args.simdir)
varnames = yd.reverse_var_alias

if args.var is None:
    av_v = sorted(set(vv for vv, *_ in sim.scrape.debug_data_keys().keys()))
    print("Available vars:")
    max_len = max(len(vv) for vv in av_v)
    for vv in av_v:
        if varnames.get(vv): print(f"  {vv.ljust(max_len)} -> {varnames[vv]}")
        else: print(f"  {vv}")
    exit(0)

func = eval(args.func)

fig, ax = plt.subplots(1, figsize=(6, 4), animated=True)

if (args.boundary is not None):
    ax.set_xlim(-args.boundary, args.boundary)
    ax.set_ylim(-args.boundary, args.boundary)

kwargs = dict(
    var=args.var,
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

plot = yp.NativeColorPlot(sim=sim, **kwargs)
if args.time_min is not None:
    args.time_min = plot.formatter.inverse_convert_time(args.time_min)

if args.time_max is not None:
    args.time_max = plot.formatter.inverse_convert_time(args.time_max)

times = plot.data.time_range
if args.time_min is not None:
    times = times[times>=args.time_min]
if args.time_max is not None:
    times = times[times<=args.time_max]
times = times[::args.time_every]

output_dir=f"{args.outputpath}/{args.var}_{args.sampling}"
frames = yp.save_frames(
        times=times,
        fig=fig,
        plots=[plot],
        output_dir=output_dir,
        prefix=f"frame_",
        dpi=300,
        )

output_mp4 = os.path.join(output_dir, "animation.mp4")
subprocess.run([
    "ffmpeg",
    "-y",                   # overwrite if exists
    "-framerate", args.fps,
    "-i", os.path.join(output_dir, "frame_%04d.png"),
    "-pix_fmt", "yuv420p",
    "-crf", "18",           # quality (lower = better, 18–23 typical)
    output_mp4,
], check=True,
stderr=subprocess.DEVNULL,
stdout=subprocess.DEVNULL)  # suppress standard output

plt.close()
