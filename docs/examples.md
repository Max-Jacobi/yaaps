# Example Scripts

The `examples/` directory contains ready-to-use command-line scripts that demonstrate the main workflows of YAAPS.

---

## `simple_plot.py` — Single 2D Snapshot

Creates a 2D colour plot of a simulation variable at a specified time and saves it as a PNG.

```bash
python examples/simple_plot.py rho -s /path/to/sim -t 100 -n log -r xy
```

**Arguments**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `var` | — | *(required)* | Variable to plot (e.g., `rho`, `ye`). |
| `--simdir` | `-s` | `active` | Directory containing `.athdf` files. |
| `--time` | `-t` | `1e5` | Simulation time to plot. |
| `--sampling` | `-r` | `xy` | Plane to plot (`xy`, `xz`, `yz`, or e.g. `x1v,x2v`). |
| `--cmap` | `-c` | — | Matplotlib colormap name. |
| `--norm` | `-n` | — | Normalization: `log` or `lin`. |
| `--boundary` | `-b` | — | Symmetric plot boundary in code units. |
| `--meshblocks` | `-m` | `False` | Draw mesh-block boundary rectangles. |
| `--func` | `-f` | `None` | Python expression evaluated to a transform function applied to the data. |
| `--vmin` | — | — | Colorscale minimum. |
| `--vmax` | — | — | Colorscale maximum. |
| `--outputpath` | `-o` | *(tmpfile)* | Path to save the PNG. |
| `--paper-format` | `-p` | `False` | Use LaTeX labels and physical units. |

Run without arguments to list available variables:

```bash
python examples/simple_plot.py
```

---

## `simple_anim.py` — Animated Frames

Renders a sequence of 2D frames and saves them as individual PNGs in a directory.

```bash
python examples/simple_anim.py rho -s /path/to/sim -b 200 --time_min 0 --time_max 50
```

**Arguments** (extends `simple_plot.py`)

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--time_min` | — | — | Start time. |
| `--time_max` | — | — | End time. |
| `--time_every` | — | `1` | Use every Nth available output time. |

Output frames are written to a directory named `{var}_{sampling}/` with filenames `frame_0000.png`, `frame_0001.png`, etc.

---

## `double_anim.py` — Side-by-Side Animation

Like `simple_anim.py` but renders two simulations side-by-side in a single figure.

```bash
python examples/double_anim.py rho -s1 /path/to/sim_A -s2 /path/to/sim_B -b 200
```

**Additional Arguments**

| Argument | Short | Description |
|----------|-------|-------------|
| `--simdir_1` | `-s1` | First simulation directory. |
| `--simdir_2` | `-s2` | Second simulation directory. |

---

## `parallel_anim.py` — History / Time-Series Plots

Renders a range of simulation snapshots to individual PNG files using
multiple worker processes:

```bash
python examples/parallel_anim.py rho \
    -s /path/to/sim \
    -o frames/ \
    -w 8 \
    -n log --vmin 1e10 --vmax 1e15
```

Each worker loads the simulation once at startup and then renders
its assigned frames independently.  Frames can afterwards be assembled
into a video with, e.g., `ffmpeg`:

```
ffmpeg -framerate 24 -i frames/frame_%06d.png -c:v libx264 out.mp4
```

**Arguments**

Same as simple_anim.py

---

## `plot_hst.py` — History / Time-Series Plots

Plots columns from `.hst`, tracer (`tra`), waveform (`wav`), or horizon (`hor`) files against time for one or more simulations.

```bash
python examples/plot_hst.py max_rho -s /path/to/sim_A /path/to/sim_B
```

**Arguments**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `vars` | — | `max_rho` | Variables to plot. Multiple variables create a grid of subplots. Comma-separated values in a single entry are overlaid on one subplot. Prefix with `hor/`, `tra/`, or `wav/` to disambiguate sources. |
| `--simdir` | `-s` | `active` | One or more simulation directories. Multiple produce overlaid lines per subplot. |
| `--xvar` | `-v` | `time` | Quantity for the x-axis. |
| `--colors` | `-c` | auto | Line colours for each simulation. |
| `--funcs` | `-f` | — | Transform functions in the form `var:expr`, e.g., `max_rho:lambda d: d*1e-14`. |
| `--ylog` | — | — | Variables to put on a log y-axis. |
| `--ylim` | — | — | Y-axis limits as `var:min:max`. |
| `--xlog` | — | `False` | Log-scale the x-axis. |
| `--xlim` | — | — | X-axis limits `min max`. |
| `--no-auto-log` | — | `False` | Disable automatic log scaling for mass and neutrino quantities. |
| `--horizon_ind` | `-a` | `0` | Horizon index for horizon quantities. |
| `--tracker_ind` | `-r` | `1` | Tracker index for tracer quantities. |
| `--wave_rad` | `-w` | `200` | Extraction radius for waveform quantities. |
| `--outputpath` | `-o` | *(tmpfile)* | Path to save the PNG. |

**Example — overlay multiple variables**

```bash
python examples/plot_hst.py "max_rho,max_ye" "m_ej" -s /path/to/sim
```

---

## `diff_inputs.py` — Compare Parameter Files

Prints a side-by-side diff of two Athena++ input or restart files, highlighting changed parameters.

```bash
python examples/diff_inputs.py run_A.par run_B.par
python examples/diff_inputs.py run_A.par run_B.par --float_tol 1e-6
python examples/diff_inputs.py run_A.par run_B.par --ignore output surface
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `input1` | *(required)* | First parameter file. |
| `input2` | *(required)* | Second parameter file. |
| `--float_tol` | `1e-8` | Relative tolerance for float comparisons. |
| `--ignore` | `output surface psi4_extraction hst_windowed` | Section prefixes to skip. |

---

## Programmatic Usage Examples

### Minimal plot

```python
import matplotlib.pyplot as plt
import yaaps as ya

sim = ya.Simulation("/path/to/sim")
sim.plot2d(time=100.0, var="rho", norm="log", sampling="xy")
plt.savefig("rho.png", dpi=200, bbox_inches="tight")
```

### Derived variable plot

```python
import numpy as np
import yaaps.plot2D as yp

def velocity_magnitude(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)

plot = yp.DerivedColorPlot(sim, var="|v|",
                            depends=("util_x", "util_y", "util_z"),
                            definition=velocity_magnitude,
                            sampling="xy", norm="log")
plot.plot(100.0)
```

### Animation to file

```python
import numpy as np
import yaaps.plot2D as yp

plot = yp.NativeColorPlot(sim, var="rho", sampling="xy", norm="log")
times = plot.data.time_range[::5]
frames = yp.save_frames(times, fig=plt.gcf(), plots=[plot],
                         output_dir="rho_frames", dpi=300)
```

### Paper-ready figure

```python
from yaaps.plot_formatter import PlotFormatter
import yaaps.plot2D as yp

formatter = PlotFormatter("paper")
plot = yp.NativeColorPlot(sim, var="rho", formatter=formatter)
# time supplied in milliseconds, converted back to code units automatically
time_code = formatter.inverse_convert_time(4.93)
plot.plot(time_code)
```

---

## See Also

- [`Simulation`](simulation.md)
- [`plot2D`](plot2D.md)
- [`plot_formatter`](plot_formatter.md)
- [`input`](input.md)
