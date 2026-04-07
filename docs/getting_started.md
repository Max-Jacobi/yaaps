# Getting Started

This guide walks you through installing YAAPS, loading your first simulation, and producing your first plot.

---

## Requirements

- Python ≥ 3.12
- A GRAthena++ simulation output directory containing:
  - At least one `.inp` or `.par` parameter file
  - `.athdf` volume-data files (for 2D plots)
  - A `.hst` history file (for time-series plots)

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Max-Jacobi/yaaps.git
cd yaaps
pip install -e .
```

This installs YAAPS and all its dependencies (`numpy`, `scipy`, `h5py`, `matplotlib`, `tqdm`, `simtroller`).

---

## 1. Load a Simulation

The [`Simulation`](simulation.md) class is the single entry point for all data access. Point it at your simulation output directory:

```python
from yaaps import Simulation

sim = Simulation("/path/to/simulation/output")

print(sim.name)        # short name derived from the directory
print(sim.problem_id)  # value of job/problem_id from the parameter file
print(sim.dx)          # finest-level grid spacings [dx1, dx2, dx3]
```

YAAPS automatically finds the `.inp` / `.par` parameter file in the directory. If the file lives elsewhere, pass it explicitly:

```python
sim = Simulation("/path/to/output", input_path="/path/to/params.par")
```

---

## 2. Inspect History Data

The `.hst` property loads the GRAthena++ history file and returns a dictionary of 1-D NumPy arrays:

```python
hst = sim.hst

print(list(hst.keys()))   # all available columns, e.g. ['time', 'dt', 'mass', ...]

import matplotlib.pyplot as plt

plt.plot(hst["time"], hst["max_rho"])
plt.xlabel("time [code]")
plt.ylabel("max density [code]")
plt.yscale("log")
plt.tight_layout()
plt.savefig("max_rho.png", dpi=150)
```

---

## 3. Make a Quick 2D Snapshot

Use `sim.plot2d` to create a colour plot of any stored variable at a given simulation time:

```python
import matplotlib.pyplot as plt
from yaaps import Simulation

sim = Simulation("/path/to/output")

fig, ax = plt.subplots(figsize=(5, 4))
sim.plot2d(time=100.0, var="rho", sampling="xy", norm="log", ax=ax)
plt.savefig("rho_xy.png", dpi=200, bbox_inches="tight")
```

**Common keyword arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `var` | Variable name or alias | `"rho"`, `"ye"`, `"hydro.prim.p"` |
| `sampling` | Slice plane | `"xy"`, `"xz"`, `"yz"` |
| `norm` | Colour scale | `"log"`, `"lin"`, `"asinh"` |
| `cmap` | Matplotlib colormap | `"magma"`, `"viridis"` |
| `vmin` / `vmax` | Colour scale limits | `1e10`, `1e15` |
| `draw_meshblocks` | Show AMR block boundaries | `True` |
| `formatter` | Label style | `"raw"` (default) or `"paper"` |

Run the `simple_plot.py` script directly from the command line for the same result:

```bash
python examples/simple_plot.py rho -s /path/to/output -t 100 -n log -r xy
```

---

## 4. Publication-Ready Labels

Switch to `formatter="paper"` to get LaTeX axis labels and physical units (km, ms, g cm⁻³, …):

```python
from yaaps.plot_formatter import PlotFormatter

formatter = PlotFormatter("paper")

# Convert a physical time (ms) back to code units before plotting
time_code = formatter.inverse_convert_time(4.93)   # ms → code

sim.plot2d(time=time_code, var="rho", sampling="xy", norm="log", formatter=formatter)
```

---

## 5. Animate Over Time

```python
import numpy as np
import matplotlib.pyplot as plt
import yaaps.plot2D as yp
from yaaps import Simulation

sim = Simulation("/path/to/output")

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)

plot = yp.NativeColorPlot(sim, var="rho", sampling="xy", norm="log", ax=ax)

# Pick every 5th available snapshot
times = plot.data.time_range[::5]

frames = yp.save_frames(times, fig=fig, plots=[plot],
                         output_dir="rho_frames", dpi=200)
print(f"Saved {len(frames)} frames to rho_frames/")
```

Or produce a video directly:

```python
anim = plot.animate(times)
anim.save("rho.mp4", fps=15)
```

---

## 6. Available Variables

To see which variables are present in your simulation:

```python
keys = sim.scrape.debug_data_keys().keys()
variables = sorted(set(var for var, *_ in keys))
for v in variables:
    print(v)
```

Short aliases (e.g. `"rho"` for `"hydro.prim.rho"`) are listed in the [Decorations](decorations.md#variable-aliases) reference.

---

## Next Steps

| Topic | Documentation |
|-------|--------------|
| Full `Simulation` API | [simulation.md](simulation.md) |
| Native / Derived / Vector data loaders | [datatypes.md](datatypes.md) |
| All plot classes and animation helpers | [plot2D.md](plot2D.md) |
| Label formatting and unit conversion | [plot_formatter.md](plot_formatter.md) |
| Unit conversion factors and LaTeX labels | [units.md](units.md) |
| Variable aliases and colormap defaults | [decorations.md](decorations.md) |
| Metric tensor utilities for derived quantities | [recipes2D.md](recipes2D.md) |
| Reading and comparing parameter files | [input.md](input.md) |
| Parallel frame rendering | [parallel_utils.md](parallel_utils.md) |
| CLI scripts reference | [examples.md](examples.md) |
