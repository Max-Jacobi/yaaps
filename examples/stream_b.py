"""Minimal StreamPlot of the B field (Bcc1, Bcc2, Bcc3)."""

import matplotlib.pyplot as plt

from yaaps import Simulation
from yaaps.datatypes import Vector
from yaaps.plot2D import StreamPlot

sim = Simulation("path/to/simulation")

# xy-plane vector from the three native B components
b_field = Vector.from_native(sim, ("Bcc1", "Bcc2", "Bcc3"), sampling=("x1v", "x2v"))

fig, ax = plt.subplots(figsize=(5, 5))
sp = StreamPlot(b_field, bounds=20.0, N_points=40, ax=ax)
sp.plot(time=0.0)

plt.savefig("stream_b.png", dpi=200, bbox_inches="tight")
