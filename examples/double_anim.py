import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import yaaps as ya
import yaaps.plot2D as yp

ap = argparse.ArgumentParser("Make an animation and save the frames as png")

ap.add_argument('var', type=str, help="Variable to plot")
ap.add_argument('-o','--outputpath', type=str, default=None,
                help="Path to save at")
ap.add_argument('-s1','--simdir_1', type=str,
                help="First simulation directory")
ap.add_argument('-s2','--simdir_2', type=str,
                help="Second simulation directory")
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

if len(sys.argv) == 1:
    sim = ya.Simulation("active")
    av_v = sorted(list(set(vv for vv, *_ in sim.scrape.debug_data_keys().keys())))
    print("Available vars:")
    for vv in av_v:
        print(f"  {vv}")
    exit(0)

args = ap.parse_args()
sims = (ya.Simulation(args.simdir_1),
        ya.Simulation(args.simdir_2))
func = eval(args.func)

fig, axs = plt.subplots(2, figsize=(6, 8), animated=True)

for ax in axs:
    ax.set_xlim(-args.boundary, args.boundary)
    ax.set_ylim(-args.boundary, args.boundary)

kwargs = dict(
    var=args.var,
    norm=args.norm,
    func=func,
    sampling=args.sampling,
    cmap=args.cmap,
    draw_meshblocks=args.meshblocks,
    vmin=args.vmin,
    vmax=args.vmax,
    format="paper" if args.paper_format else "raw",
    )

for k, v in list(kwargs.items()):
    if v is None:
        kwargs.pop(k)

plots = [yp.NativeColorPlot(sim=sim, ax=ax, **kwargs)
         for ax, sim in zip(axs, sims)]

times = np.unique(np.concatenate([p.data.time_range for p in plots]))
if args.time_min is not None:
    times = times[times>=args.time_min]
if args.time_max is not None:
    times = times[times<=args.time_max]
times = times[::args.time_every]

frames = yp.save_frames(
        times=times,
        fig=fig,
        plots=plots,
        output_dir=f"{args.var}_{args.sampling}",
        prefix=f"frame_",
        dpi=300,
        )

plt.close()
