import os
import sys
from math import isqrt
import tempfile
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaaps as ya

from typing import Callable

auto_log_keys = [
    'mass',
    'max_sc_nG_00', 'max_sc_nG_01', 'max_sc_nG_02',
    'max_sc_E_00', 'max_sc_E_01', 'max_sc_E_02',
    'max_sc_n_00', 'max_sc_n_01', 'max_sc_n_02',
    'max_sc_J_00', 'max_sc_J_01', 'max_sc_J_02',
    ]

ap = argparse.ArgumentParser("Create a 2D grid plot using yaaps and save it as png")

ap.add_argument('vars', type=str, help="Variables to plot. "
                "Optionally the source can be specified as hor/var, tra/var, wav/var to avoid ambiguity. "
                "Multiple variables can be plotted in the same subplot by separating them with commas.",
                nargs='+', default=["max_rho"])
ap.add_argument('-o','--outputpath', type=str, default=None,
                help="Path to save at")
ap.add_argument('-t','--time', type=float, default=1e5,
                help="Time to plot at")
ap.add_argument('-s','--simdir', type=str, default=['active'], nargs='+',
                help="Directories to look for athdf files in")
ap.add_argument('-c','--colors', type=str, default=None, nargs='+',
                help="Colors to plot the simulations in")
ap.add_argument('-v','--xvar', type=str, default="time",
                help="Quantity to plot on the x axis")
ap.add_argument('-a', '--horizon_ind', type=int, default=0,
                help="Index of the horizon to plot (for horizon quantities).")
ap.add_argument('-r', '--tracker_ind', type=int, default=1,
                help="Index of the tracker to plot (for tracker quantities).")
ap.add_argument('-w', '--wave_rad', type=float, default=200,
                help="Index of the wave surface to plot (for wave quantities).")
ap.add_argument('-f','--funcs', type=str, nargs='+', default=[],
                help="Modify plot with given functions in the form var:func (calls eval).")
ap.add_argument('--ylog', type=str, nargs='+', default=[],
                help="Vars to log scale the yaxis on")
ap.add_argument('--ylim', type=str, nargs='+', default=[],
                help="Combination of keys and respective limits for y axis, in the form var:min:max")
ap.add_argument('--xlog', type=bool, default=False,
                help="Log scale the xaxis")
ap.add_argument('--xlim', type=float, nargs=2, default=None,
                help="Limits for x axis")
ap.add_argument('--no-auto-log', action='store_true',
                help="Disable automatic log scaling for y axis")

args = ap.parse_args()

sims = []
for sim in args.simdir:
    try:
        sims.append(ya.Simulation(sim))
    except FileNotFoundError:
        print(f"No parfile in {sim}, skipping", file=sys.stderr)

vars = []
for v in args.vars:
    if ',' in v:
        vars.append(v.split(','))
    else:
        vars.append(v)


if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

if args.colors is None:
    args.colors = [f"C{ii}" for ii, _ in enumerate(sims)]
elif len(sims) > len(args.colors):
    raise ValueError("Not enough colors for simulations")

def eval_f(f: str) -> Callable:
    if f in ['None', 'id']:
        return lambda d: d
    if f == 'relabs':
        return lambda d: np.abs(d/d[0] - 1)
    if f == 'inv':
        return lambda d: 1/d

    func = eval(f)
    if isinstance(func, (int, float)):
        return lambda d: d*func

    if not callable(func):
        raise ValueError(f"{f} does not evaluate to a callable object")
    return func

funcs = {var.strip(): eval_f(f.strip()) for var, f in map(lambda s: s.split(':'), args.funcs)}

ylim_dict = {}
for item in args.ylim:
    var, mn, mx = item.split(':')
    ylim_dict[var] = (float(mn), float(mx))

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

def plot(var, ax, sim, **kw):
    if "/" in var:
        src_label, var = var.split("/", 1)
        src = sources(sim).get(src_label)
        if src is None or var not in src:
            raise KeyError(f"{var} not found in {src_label}")
    else:
        for src_label, src in sources(sim).items():
            if var in src:
                break
        else:
            raise KeyError(f"{var} not found")

    data = src[var]

    if f'{src_label}/{var}' in funcs:
        data = funcs[f'{src_label}/{var}'](data)
    elif var in funcs:
        data = funcs[var](data)

    return ax.plot(src[args.xvar], data, **kw)

m, n = split(len(vars))
fig, axs = plt.subplots(m, n, figsize=(n*7, m*4), sharex=True)
axs = np.atleast_1d(axs)

if len(sims) > 1:
    common_path = os.path.commonpath([sim.path for sim in sims])
else:
    common_path = os.path.dirname(os.path.dirname(sims[0].path))

suffix = ''
bases = [os.path.basename(sim.path) for sim in sims]
if all(b == bases[0] for b in bases):
    suffix = '/' + bases[0]


for var, ax in zip(vars, axs.flat):
    for sim, c in zip(sims, args.colors):
        name = sim.path.replace(common_path, '').strip('/').replace(suffix, '')
        if isinstance(var, list):
            for v, ls in zip(var, ('-', '--', ':', '-.')):
                if sim.path == sims[0].path:
                    label = v
                else:
                    label= None
                plot(v, ax, sim, c=c, ls=ls, label=label)
        else:
            plot(var, ax, sim, c=c, label=name)
    ax.set_xlabel(args.xvar)

    if isinstance(var, list):
        ax.set_ylabel(" ".join(var))
        for v in var:
            if v in args.ylog + (auto_log_keys if not args.no_auto_log else []):
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
    elif not sim_legend_exists:
        ax.legend()
        sim_legend_exists = True
    if isinstance(var, list):
        for v in var:
            if v in ylim_dict:
                ax.set_ylim(ylim_dict[v])
    elif var in ylim_dict:
        ax.set_ylim(ylim_dict[var])

if args.xlog:
    for ax in axs.flat:
        ax.set_xscale('log')
if args.xlim is not None:
    for ax in axs.flat:
        ax.set_xlim(args.xlim)

plt.tight_layout()
plt.savefig(args.outputpath, dpi=200, bbox_inches='tight')
print(args.outputpath)
