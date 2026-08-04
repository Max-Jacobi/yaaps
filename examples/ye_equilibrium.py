"""Local Y_e vs local weak-equilibrium Y_e from the M1 output.

The equilibrium expression itself lives in yaaps.recipes2D.ye_equilibrium,
which also documents its derivation from the GR-Athena++ M1 sources.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import yaaps as ya
import yaaps.decorations as yd
import yaaps.plot2D as yp
from yaaps.recipes2D import ye_equilibrium

# species indices: NUE=0, NUA(nuebar)=1, NUX=2
DEPENDS = (
    "passive_scalar.r_0",  # Y_e
    "M1.radmat.sc_eta_0_00",  # nue number emissivity
    "M1.radmat.sc_kap_a_0_00",  # nue number absorption opacity
    "M1.rad.sc_n_00",  # densitized nue number density
    "M1.radmat.sc_eta_0_01",
    "M1.radmat.sc_kap_a_0_01",
    "M1.rad.sc_n_01",
    "geom.adm.gxx",
    "geom.adm.gxy",
    "geom.adm.gxz",
    "geom.adm.gyy",
    "geom.adm.gyz",
    "geom.adm.gzz",
)

ap = argparse.ArgumentParser("Plot Y_e next to the local weak-equilibrium Y_e")
ap.add_argument(
    "-s", "--simdir", type=str, default="active", help="Directory to look for athdf files in"
)
ap.add_argument(
    "-o", "--outputpath", type=str, default="ye_equilibrium.png", help="Path to save at"
)
ap.add_argument("-t", "--time", type=float, default=None, help="Time to plot at (default: last)")
ap.add_argument("-r", "--sampling", type=str, default="xz", help="Plane to plot")
ap.add_argument("-b", "--boundary", type=float, default=100.0, help="Plot boundary in km")
ap.add_argument("--vmin", type=float, default=0.0, help="Minimum of the colorscale")
ap.add_argument("--vmax", type=float, default=0.6, help="Maximum of the colorscale")
ap.add_argument(
    "--kap-eff-min",
    type=float,
    default=1e-6,
    help="Mask cells with effective interaction opacity below this (0 disables)",
)
args = ap.parse_args()

sim = ya.Simulation(args.simdir)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

style = dict(
    sampling=args.sampling,
    formatter="paper",
    cmap=yd.ye_cmap,
    norm="lin",
    vmin=args.vmin,
    vmax=args.vmax,
)

p_ye = yp.NativeColorPlot(sim, var="ye", ax=axs[0], **style)
p_eq = yp.DerivedColorPlot(
    sim,
    "ye_eq",
    depends=DEPENDS,
    definition=lambda *dep: ye_equilibrium(*dep, kap_eff_min=args.kap_eff_min),
    ax=axs[1],
    **style,
)
p_eq.formatter.field_labels.add_label("ye_eq", r"$Y_e^{\rm eq}$")

time = p_eq.data.time_range[-1] if args.time is None else args.time
for p in (p_ye, p_eq):
    p.plot(time)

for ax in axs:
    ax.set_xlim(-args.boundary, args.boundary)
    ax.set_ylim(-args.boundary, args.boundary)

fig.savefig(args.outputpath, dpi=200)
print(f"{args.outputpath} at t_code = {time}")
