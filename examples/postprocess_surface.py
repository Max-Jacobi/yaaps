################################################################################
import argparse
import os
from datetime import datetime
from typing import Any

import numpy as np

from yaaps import Simulation
from yaaps.plot2D import Derived
from yaaps.surface import Surfaces
from yaaps.surface import surface_func as sf

################################################################################

parser = argparse.ArgumentParser(
    description="Postprocessing for surface outputs"
)
parser.add_argument("-E", "--eos", default=None, type=str,
                    help="EOS table in pycompose format")
parser.add_argument("simpath", type=str, help="Path to simulation")
parser.add_argument("-o", "--outputpath", default=None,
                    help="Directory to output to")
parser.add_argument("-s", "--isurf", default=1, type=int,
                    help="Index of surface to use")
parser.add_argument("-r", "--irad", default=0, type=int,
                    help="Index of radius to use")
parser.add_argument("-c", "--criteria", nargs='*', default=["bernoulli_out"], type=str,
                    help=("Ejection criteria to compute."
                          " Choices: geodesic, bernoulli, "
                          "geodesic_out, bernoulli_out, none, none_out"))
parser.add_argument("-g", "--histograms",  nargs='*', default=[], type=str,
                    help=("Ejecta histograms to calculate. "
                          " Choices: vinf or any dataset name in the files"))
parser.add_argument("-w", "--weighted_averages",  nargs='*', default=[], type=str,
                    help=("Weighted average time series to calculate. "
                          " Choices: vinf or any dataset name in the files"))
parser.add_argument("-m", "--mass_ejection", action="store_true",
                    help="Calculate masse ejection rate")
parser.add_argument("-l", "--nu_luminosities", action="store_true",
                    help="Calculate neutrino luminosities")
parser.add_argument("-e", "--nu_energies", action="store_true",
                    help="Calculate neutrino energies")
parser.add_argument("-b", "--backtrack_temp", type=float, default=8.0,
                    help="Temperature for backtracking in GK. Default=8")
parser.add_argument("-n", "--ncpu", default=1, type=int,
                    help="Number of cores to use")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Print progress")
args = parser.parse_args()

if args.outputpath is None:
    today = datetime.today().strftime('%Y-%m-%d')
    args.outputpath = f"{args.simpath}/surface_analysis_s{args.isurf}_r{args.irad}_{today}"

os.makedirs(args.outputpath, exist_ok=True)


################################################################################

paths = [f"{args.simpath}/{d}" for d in os.listdir(args.simpath)
         if (os.path.isdir(f"{args.simpath}/{d}")
             and d.startswith('output-'))]

temp_bt = args.backtrack_temp / 11.60452

s = Surfaces(
    paths,
    args.isurf,
    args.irad,
    n_cpu=args.ncpu,
    verbose=args.verbose,
    eos_path=args.eos,
    )
dt = s.times[1] - s.times[0]

################################################################################

bins = {
    "passive_scalars.r_0": np.linspace(0, 0.65, 66),
    "vinf": np.linspace(0, 1, 101),
    "hydro.aux.s": np.linspace(0, 250, 101),
    "hydro.aux.T": np.linspace(0, 1, 101),
    "tau": np.geomspace(20.3, 20300., 101),
    "tau_b": np.geomspace(20.3, 20300., 101),
    "ph": np.linspace(0, 2*np.pi, len(s.aux['ph'])+1),
    "th": np.linspace(0, np.pi, len(s.aux['th'])+1),
    }

################################################################################

def _check_extra(f: str, crit: str) -> str | sf.SurfaceFunc:
        if f == 'vinf':
            if 'bernoulli' in crit:
                return sf.vinf['bernoulli']
            return sf.vinf['geodesic']
        if f == 'tau':
            return sf.tau
        if f == 'tau_b':
            return sf.tau_b(temp_bt, s.eos)
        return f

surf_funcs = {}

for crit in args.criteria:
    if args.mass_ejection:
        surf_funcs[f'sc_mdot_{crit}'] = sf.mdot[crit]
    for f in args.weighted_averages:
        _f = _check_extra(f, crit)
        surf_funcs[f'sc_mdot_{f}_{crit}'] = sf.wmdot(_f)[crit]
    for f in args.histograms:
        _f = _check_extra(f, crit)
        surf_funcs[f'hist_{f}_{crit}'] = sf.hist(_f, bins=bins[f])[crit]

if args.nu_luminosities:
    for inu in range(3):
        surf_funcs[f'sc_nu{inu}_lum'] = sf.nu_lum[inu]

if args.nu_energies:
    for inu in range(3):
        surf_funcs[f'sc_nu{inu}_en'] = sf.nu_e[inu]

all_funcs = sf.get_many(surf_funcs.values())
all_funcs.name = 'surface reductions'

data: dict[str, Any] = {k: [] for k in surf_funcs}
for raw_data in s.process_h5_parallel((all_funcs,), ordered=True):
    for k, d in zip(surf_funcs.keys(), raw_data[0]):
        data[k].append(d)

for c in args.criteria:
    data[f'mej_{c}'] = np.cumsum(data[f'sc_mdot_{c}'])*dt

    for h in args.histograms:
        hist, bin_edges = zip(*data[f'hist_{h}_{c}'])
        data[f'hist_{h}_{c}_cum'] = sum(hist), bin_edges[0]


scalars = tuple(k[3:] for k in data if  k.startswith("sc_"))

for sc in scalars:
    data[f"sc_{sc}"] = np.array(data[f"sc_{sc}"])

################################################################################

scalar_headers = [sc.replace('bernoulli', 'b').replace('geodesic', 'g')
                  for sc in scalars]
header = ('{:<24s} '*(len(scalars)+1)).format("time", *scalar_headers)
sdata = np.stack([s.times] + [data[f"sc_{sc}"] for sc in scalars], axis=1)
np.savetxt(f"{args.outputpath}/scalars.txt", sdata, fmt="%24.16e", header=header)

################################################################################

hists = tuple(k[5:] for k in data if k.startswith("hist_") and not k.endswith('_cum'))
for h in hists:
    hist, bin_edges = data[f'hist_{h}_cum']
    with open(f"{args.outputpath}/hist_{h}.txt", 'w') as hf:
        hf.writelines((" ".join(bin_edges.astype(str)) + "\n",
                       " ".join(hist.astype(str)) + "\n"))
