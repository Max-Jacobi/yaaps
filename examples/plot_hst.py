import argparse
import os
import sys
import tempfile
from collections.abc import Callable
from inspect import signature
from math import isqrt

import matplotlib.pyplot as plt
import numpy as np

import yaaps as ya

auto_log_keys = [
    "mass",
    "max_sc_nG_00",
    "max_sc_nG_01",
    "max_sc_nG_02",
    "max_sc_E_00",
    "max_sc_E_01",
    "max_sc_E_02",
    "max_sc_n_00",
    "max_sc_n_01",
    "max_sc_n_02",
    "max_sc_J_00",
    "max_sc_J_01",
    "max_sc_J_02",
]

ap = argparse.ArgumentParser("Create a 2D grid plot using yaaps and save it as png")

ap.add_argument(
    "vars",
    type=str,
    help="Variables to plot. "
    "Optionally the source can be specified as hor/var, tra/var, wav/var to avoid ambiguity. "
    "Multiple variables can be plotted in the same subplot by separating them with commas.",
    nargs="*",
    default=["max_rho"],
)
ap.add_argument("-o", "--outputpath", type=str, default=None, help="Path to save at")
ap.add_argument(
    "-s",
    "--simdir",
    type=str,
    default=["."],
    nargs="+",
    help="Directories to look for athdf files in",
)
ap.add_argument(
    "-l", "--listvars", action="store_true", help="List the available variables and exit"
)
ap.add_argument(
    "-c", "--colors", type=str, default=None, nargs="+", help="Colors to plot the simulations in"
)
ap.add_argument("-v", "--xvar", type=str, default="time", help="Quantity to plot on the x axis")
ap.add_argument(
    "-a",
    "--horizon_ind",
    type=int,
    default=0,
    help="Index of the horizon to plot (for horizon quantities).",
)
ap.add_argument(
    "-r",
    "--tracker_ind",
    type=int,
    default=1,
    help="Index of the tracker to plot (for tracker quantities).",
)
ap.add_argument(
    "-w",
    "--wave_rad",
    type=float,
    default=200,
    help="Index of the wave surface to plot (for wave quantities).",
)
ap.add_argument(
    "-f",
    "--funcs",
    type=str,
    nargs="+",
    default=[],
    help="Modify plot with given functions in the form var:func (calls eval).",
)
ap.add_argument("--ylog", type=str, nargs="+", default=[], help="Vars to log scale the yaxis on")
ap.add_argument(
    "--ylim",
    type=str,
    nargs="+",
    default=[],
    help="Combination of keys and respective limits for y axis, in the form var:min:max",
)
ap.add_argument("--xlog", type=bool, default=False, help="Log scale the xaxis")
ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Limits for x axis")
ap.add_argument(
    "--no-auto-log", action="store_true", help="Disable automatic log scaling for y axis"
)
ap.add_argument("--no-legend", action="store_true", help="Disable legend for simulations")

args = ap.parse_args()

sims = []
for sim in args.simdir:
    try:
        sims.append(ya.Simulation(sim))
    except FileNotFoundError:
        print(f"No parfile in {sim}, skipping", file=sys.stderr)

if args.listvars:
    print("Available vars in sims[0]:")
    print(sims[0].hst.keys())
    exit(0)

vars = []
for v in args.vars:
    if "," in v:
        vars.append(v.split(","))
    else:
        vars.append(v)


if args.outputpath is None:
    fd, args.outputpath = tempfile.mkstemp(suffix=".png")
    os.close(fd)

if args.colors is None:
    args.colors = [sim.md.get("color", f"C{i}") for i, sim in enumerate(sims)]

elif len(sims) > len(args.colors):
    raise ValueError("Not enough colors for simulations")


def _diff_eq_len(a, b):
    r = np.zeros_like(a)
    r[:-1] = np.diff(a) / np.diff(b)
    return r


def _test_float(f: str) -> bool:
    try:
        float(f)
        return True
    except ValueError:
        return False


def eval_f(f: str) -> Callable:
    if _test_float(f):
        f = float(f)
        return lambda d: d * f
    if f in ["None", "id"]:
        return lambda d: d
    elif f in ("relabs", "absrel"):
        return lambda d: np.abs(d / d[0] - 1)
    elif f == "absdiff":
        return lambda d: np.abs(d - d[0])
    elif f == "diff":
        return lambda d: d - d[0]
    elif f == "inv":
        return np.reciprocal
    elif f == "abs":
        return np.abs
    elif f == "ddt":
        return _diff_eq_len

    func = eval(f)
    if isinstance(func, (int, float)):
        return lambda d: d * func

    if not callable(func):
        raise ValueError(f"{f} does not evaluate to a callable object")
    return func


def apply_func(func: Callable, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(signature(func).parameters) == 1:
        return func(data)
    return func(data, x)


func_names = {var: f for var, f in map(lambda s: s.split(":"), args.funcs)}
funcs = {var: eval_f(f) for var, f in func_names.items()}


def func_label(var: str) -> str:
    f = func_names.get(var, "None")
    if f in ["None", "id"]:
        return var
    if f in ("relabs", "absrel"):
        return f"|{var}/ {var}[0] - 1|"
    if f == "absdiff":
        return f"|{var} - {var}[0]|"
    if f == "diff":
        return f"{var} - {var}[0]"
    if f == "inv":
        return f"1/{var}"
    if f == "abs":
        return f"|{var}|"
    if f == "ddt":
        return f"d{var}/dt"
    if f.startswith("lambda"):
        return f"λ({var})"
    if f.startswith("np."):
        return f"{f[3:]}({var})"
    if f.startswith("math."):
        return f"{f[5:]}({var})"
    if _test_float(f):
        return f"{float(f):.2e}*{var}"
    else:
        return f"{f}({var})"


ylim_dict = {}
for item in args.ylim:
    var, mn, mx = item.split(":")
    ylim_dict[var] = (float(mn), float(mx))


def split(N):
    for n in range(isqrt(N), 0, -1):
        if not (N % n):
            return n, N // n
    raise ValueError


def sources(sim):
    src = {"hst": sim.hst}
    for s, a in zip(
        ("horizon", "tra", "wav"), (args.horizon_ind, args.tracker_ind, args.wave_rad), strict=False
    ):
        try:
            src[s[:3]] = getattr(sim, s)(a)
        except FileNotFoundError:
            continue
    return src


def find_source(sim, var):
    for label, src in sources(sim).items():
        if var in src:
            return label, src
    raise KeyError(f"{var} not found")


def plot(var, ax, sim, **kw):
    if "/" in var:
        src_label, var = var.split("/", 1)
        src = sources(sim).get(src_label)
        if src is None or var not in src:
            raise KeyError(f"{var} not found in {src_label}")
    else:
        src_label, src = find_source(sim, var)

    data = src[var]

    if f"{src_label}/{var}" in funcs:
        # data = funcs[f"{src_label}/{var}"](data)
        data = apply_func(funcs[f"{src_label}/{var}"], data, src[args.xvar])
    elif var in funcs:
        # data = funcs[var](data)
        data = apply_func(funcs[var], data, src[args.xvar])

    return ax.plot(src[args.xvar], data, **kw)


m, n = split(len(vars))
fig, axs = plt.subplots(m, n, figsize=(n * 7, m * 4), sharex=True)
axs = np.atleast_1d(axs)


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


for var, ax in zip(vars, axs.flat, strict=False):
    for sim, c in zip(sims, args.colors, strict=False):
        name = sim_labels[sim.path]
        try:
            if isinstance(var, list):
                for v, ls in zip(var, ("-", "--", ":", "-."), strict=False):
                    if sim.path == sims[0].path:
                        label = v
                    else:
                        label = None
                    plot(v, ax, sim, c=c, ls=ls, label=label)
            else:
                plot(var, ax, sim, c=c, label=name)
        except FileNotFoundError:
            print(f"No hst file in {sim.path}, skipping", file=sys.stderr)
        except KeyError:
            print(f"{var} not found in {sim.path}, skipping", file=sys.stderr)
    ax.set_xlabel(args.xvar)

    if isinstance(var, list):
        ylabel = " ".join(var)
        for v in var:
            if v in func_names:
                ylabel = ylabel.replace(v, func_label(v))
        ax.set_ylabel(ylabel)
        for v in var:
            if v in args.ylog + (auto_log_keys if not args.no_auto_log else []):
                ax.set_yscale("log")
                break
    else:
        ylabel = var
        if var in func_names:
            ylabel = func_label(var)
        ax.set_ylabel(ylabel)
        if var in args.ylog:
            ax.set_yscale("log")

sim_legend_exists = False
for var, ax in zip(vars, axs.flat, strict=False):
    if isinstance(var, list):
        ax.legend()
    elif not sim_legend_exists and not args.no_legend:
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
        ax.set_xscale("log")
if args.xlim is not None:
    for ax in axs.flat:
        ax.set_xlim(args.xlim)

plt.tight_layout()
plt.savefig(args.outputpath, dpi=200, bbox_inches="tight")
print(args.outputpath)
