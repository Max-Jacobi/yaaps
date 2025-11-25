import os
from itertools import zip_longest
from math import isqrt
import tempfile
import argparse
import matplotlib.pyplot as plt
import yaaps as ya

# for -f argument
import numpy as np

ap = argparse.ArgumentParser("Create a 2D grid plot using yaaps and save it as png")

ap.add_argument('vars', type=str, help="Variables to plot", nargs='+', default=["max_rho"])
ap.add_argument('-o','--outputpath', type=str, default=None,
                help="Path to save at")
ap.add_argument('-t','--time', type=float, default=1e5,
                help="Time to plot at")
ap.add_argument('-s','--simdir', type=str, default=['active'], nargs='+',
                help="Directories to look for athdf files in")
ap.add_argument('-c','--colors', type=str, default=None, nargs='+',
                help="Colors to plot the simulations in")
ap.add_argument('-v','--xval', type=str, default="time",
                help="Quantity to plot on the x axis")
ap.add_argument('-a', '--horizon_ind', type=int, default=0,
                help="Index of the horizon to plot (for horizon quantities).")
ap.add_argument('-r', '--tracker_ind', type=int, default=1,
                help="Index of the tracker to plot (for tracker quantities).")
ap.add_argument('-w', '--wave_rad', type=float, default=200,
                help="Index of the wave surface to plot (for wave quantities).")
ap.add_argument('-f','--funcs', type=str, nargs='+', default=[],
                help="Modify plot with given functions. (calls eval)")
ap.add_argument('--ylog', type=str, nargs='+', default=[],
                help="Vars to log scale the yaxis on")
ap.add_argument('--xlog', type=bool, default=False,
                help="Log scale the xaxis")

args = ap.parse_args()

sims = [ya.Simulation(sim) for sim in args.simdir]

vars = []
for v in args.vars:
    if ' ' in v:
        vars.append(v.split())
    else:
        vars.append(v)


if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

if args.colors is None:
    args.colors = [f"C{ii}" for ii, _ in enumerate(sims)]
elif len(sims) > len(args.colors):
    raise ValueError("Not enough colors for simulations")

def eval_f(f):
    if f in ['None', 'id']:
        return lambda d: d
    if f == 'relabs':
        return lambda d: np.abs(d/d[0] - 1)
    return eval(f)

funcs = [eval_f(f) for f, _ in zip_longest(args.funcs, vars, fillvalue='id')]

def split(N):
    for n in range(isqrt(N), 0, -1):
        if not (N%n): return n, N//n
    raise ValueError

def sources(sim):
    src = {'hst': sim.hst}
    for s, a in zip(("horizon", "tra", "wav"),
                    (args.horizon_ind, args.tracker_ind, args.wave_rad)):
        try:
            src[s[:3]] = getattr(sim, s)(a)
        except FileNotFoundError:
            continue
    return src

def plot(var, ax, sim, func, **kw):
    if "/" in var:
        var_type, var = var.split("/", 1)
        src = sources(sim).get(var_type)
        if src is None or var not in src:
            raise KeyError(f"{var} not found in {var_type}")
    else:
        for src in sources(sim).values():
            if var in src:
                break
        else:
            raise KeyError(f"{var} not found")

    return ax.plot(src[args.xval], func(src[var]), **kw)

m, n = split(len(vars))
fig, axs = plt.subplots(m, n, figsize=(n*10, m*4), sharex=True)
axs = np.atleast_1d(axs)

common_path = os.path.commonpath([sim.path for sim in sims])

for var, ax, func in zip(vars, axs.flat, funcs):
    for sim, c in zip(sims, args.colors):
        name = sim.path.replace(common_path, '').strip('/')
        if isinstance(var, list):
            for v, ls in zip(var, ('-', '--', ':', '-.')):
                if sim.path == sims[0].path:
                    label = v
                else:
                    label= None
                plot(v, ax, sim, func, c=c, ls=ls, label=label)
        else:
            plot(var, ax, sim, func, c=c, label=name)
    ax.set_xlabel(args.xval)

    if isinstance(var, list):
        ax.set_ylabel(" ".join(var))
        for v in var:
            if v in args.ylog:
                ax.set_yscale('log')
                break
    else:
        ax.set_ylabel(var)
        if var in args.ylog:
            ax.set_yscale('log')

sim_legend_exists = False
for var, ax in zip(vars, axs.flat):
    if isinstance(var, list):
        ax.legend()
    if not sim_legend_exists:
        ax.legend()
        sim_legend_exists = True

if args.xlog:
    for ax in axs.flat:
        ax.set_xscale('log')

plt.tight_layout()
plt.savefig(args.outputpath, dpi=200, bbox_inches='tight')
print(args.outputpath)
