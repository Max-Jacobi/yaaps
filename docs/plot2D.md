# 2D Plotting (`plot2D`)

The `plot2D` module provides all classes and functions for creating 2D visualisations of GRAthena++ simulation data. It supports colour plots, quiver (arrow) plots, streamline plots, contour plots, scatter/tracer plots, mesh-block overlays, time-bar plots, and animations.

## Import

```python
import yaaps.plot2D as yp
from yaaps.plot2D import (
    NativeColorPlot, DerivedColorPlot,
    QuiverPlot, StreamPlot, ContourPlot,
    TracerPlot, MeshBlockPlot, TimeBarPlot,
    animate, save_frames, interpolate_octree_to_grid,
)
```

---

## Class Hierarchy

```
Plot (abstract)
├── TimeBarPlot
├── MeshBlockPlot
├── ColorPlot (abstract)
│   ├── NativeColorPlot
│   └── DerivedColorPlot
├── ScatterPlot (abstract)
│   └── TracerPlot
├── QuiverPlot
├── StreamPlot
└── ContourPlot (abstract)
```

---

## Class: `Plot` (abstract)

Base class that defines the shared interface for all plot types.

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `ax` | `Axes` | Matplotlib `Axes` used by this plot. |
| `formatter` | `PlotFormatterBase` | Formatter controlling labels and unit conversions. |
| `t_off` | `float` | Time offset (default `0.0`), used as `t_merg` if set on the simulation. |

### Constructor

```python
Plot(ax, formatter=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `Axes \| None` | — | Matplotlib axes. `None` uses `plt.gca()`. |
| `formatter` | `PlotFormatterBase \| str \| None` | `None` | `"raw"`, `"paper"`, a formatter object, or `None` (defaults to `"raw"`). |

### `set_plot_mode`

```python
plot.set_plot_mode(mode)
```

Switch the formatter mode (`"raw"` or `"paper"`) without recreating the plot.

### `plot` (abstract)

```python
plot.plot(time) -> list[Artist]
```

Draw or update the plot at the given simulation time.

### `clean` (abstract)

```python
plot.clean()
```

Remove all artists created by this plot from the axes.

### `animate`

```python
plot.animate(times, **kwargs) -> FuncAnimation
```

Convenience wrapper around the module-level [`animate`](#function-animate) function.

---

## Class: `TimeBarPlot`

A vertical line that moves along the x-axis to indicate the current time on a time-series plot.

```python
TimeBarPlot(ax=None, formatter=None, **kwargs)
```

`**kwargs` are forwarded to `ax.axvline()`.

**Example**

```python
fig, (ax_map, ax_hst) = plt.subplots(2)
bar = yp.TimeBarPlot(ax=ax_hst, color='red', lw=1.5)
```

---

## Class: `MeshBlockPlot`

Draws rectangles showing the boundaries of each adaptive-mesh-refinement mesh block.

```python
MeshBlockPlot(data, ax=None, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `MeshData` | Data object providing mesh-block coordinates. |

`**kwargs` override rectangle defaults (`edgecolor='gray'`, `facecolor='none'`, `linewidth=0.5`, `alpha=0.2`).

---

## Class: `ColorPlot` (abstract)

Generic `pcolormesh`-based colour plot for any `MeshData` object.

```python
ColorPlot(data, ax=None, cbar=True, func=None, formatter=None, draw_meshblocks=False, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `MeshData` | — | Data object to visualise. |
| `ax` | `Axes \| None` | `None` | Target axes. |
| `cbar` | `Axes \| bool` | `True` | `True` → auto-create colorbar axes; an `Axes` → use it; `False` → no colorbar. |
| `func` | `Callable \| None` | `None` | Optional function applied to data before plotting. |
| `formatter` | `PlotFormatterBase \| str \| None` | `None` | Label/unit formatter. |
| `draw_meshblocks` | `bool` | `False` | Overlay mesh-block boundaries. |
| `**kwargs` | | | Extra keyword arguments forwarded to `pcolormesh`. |

The axes aspect ratio is set to `'equal'` automatically and axis labels are applied from the formatter.

---

## Class: `NativeColorPlot`

Colour plot for a native (directly stored) variable. Convenience subclass of `ColorPlot`.

```python
NativeColorPlot(sim, var, sampling=('x1v', 'x2v'), t_merg_offset=True, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | `Simulation` | — | Parent simulation object. |
| `var` | `str` | — | Variable name or alias (e.g., `"rho"`). |
| `sampling` | `Sampling` | `('x1v', 'x2v')` | Coordinate sampling. |
| `t_merg_offset` | `bool` | `True` | If `True` and `sim.md["t_merg"]` exists, subtract it from displayed times. |

**Example**

```python
import matplotlib.pyplot as plt
import yaaps as ya
import yaaps.plot2D as yp

sim = ya.Simulation("/path/to/sim")
fig, ax = plt.subplots()
plot = yp.NativeColorPlot(sim, var="rho", sampling="xy", norm="log", ax=ax)
plot.plot(100.0)
plt.show()
```

---

## Class: `DerivedColorPlot`

Colour plot for a derived (computed) variable. Convenience subclass of `ColorPlot`.

```python
DerivedColorPlot(sim, var, depends, definition, sampling=('x1v', 'x2v'), t_merg_offset=True, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | `Simulation` | — | Parent simulation object. |
| `var` | `str` | — | Name for the derived variable. |
| `depends` | `tuple[str, ...]` | — | Variable names that the definition depends on. |
| `definition` | `Callable` | — | Function computing the derived quantity from its dependencies. |
| `sampling` | `Sampling` | `('x1v', 'x2v')` | Coordinate sampling. |
| `t_merg_offset` | `bool` | `True` | Apply merger time offset if available. |

**Example**

```python
import numpy as np

def vmag(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)

plot = yp.DerivedColorPlot(sim, var="|v|",
                            depends=("util_x", "util_y", "util_z"),
                            definition=vmag, sampling="xy")
plot.plot(100.0)
```

---

## Class: `TracerPlot`

Scatter plot for tracer particle positions, with optional colour mapping and trailing lines.

```python
TracerPlot(tracers, coord_keys=('x1', 'x2'), color_key=None,
           trail_len=0, line_kwargs={}, formatter=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracers` | `list[dict]` | — | List of tracer data dictionaries containing `'time'` and coordinate keys. |
| `coord_keys` | `tuple[str, str]` | `('x1', 'x2')` | Keys for x and y coordinate arrays inside each tracer dict. |
| `color_key` | `str \| None` | `None` | Key for colour-mapping each point. |
| `trail_len` | `float` | `0` | If > 0, draw a trailing line of this time duration behind each particle. |
| `line_kwargs` | `dict` | `{}` | Keyword arguments for the trailing `Line2D` objects. |
| `**kwargs` | | | Forwarded to `ax.scatter()`. |

**Example**

```python
tracers = [sim.tra(i) for i in range(5)]
tp = yp.TracerPlot(tracers, trail_len=10.0, s=5, color="C1")
tp.plot(100.0)
```

---

## Class: `QuiverPlot`

Arrow (quiver) plot for a 2-D vector field.

```python
QuiverPlot(data, bounds, N_arrows=20, ax=None, grid_type="cartesian",
           func=None, formatter=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Vector` | — | [`Vector`](datatypes.md#class-vector) data object. |
| `bounds` | `float \| tuple` | — | Single float → symmetric `(-b, b, -b, b)`; 4-tuple `(x_min, x_max, y_min, y_max)` for Cartesian; `(r_min, r_max, φ_min, φ_max)` for polar. |
| `N_arrows` | `int \| tuple[int, int]` | `20` | Number of arrows per dimension. |
| `grid_type` | `str` | `"cartesian"` | `"cartesian"` or `"polar"`. |
| `func` | `Callable \| None` | `None` | Optional function applied to `(u, v)` before plotting. |
| `**kwargs` | | | Forwarded to `ax.quiver()`. |

**Example**

```python
from yaaps.datatypes import Vector
vel = Vector.from_native(sim, ("util_x", "util_y", "util_z"))
qp = yp.QuiverPlot(vel, bounds=100.0, N_arrows=15)
qp.plot(100.0)
```

---

## Class: `StreamPlot`

Streamline plot for a 2-D vector field.

```python
StreamPlot(data, bounds, N_points=20, ax=None, formatter=None, t_merg_offset=True, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Vector` | — | [`Vector`](datatypes.md#class-vector) data object. |
| `bounds` | `float \| tuple` | — | Domain bounds (same format as `QuiverPlot`). |
| `N_points` | `int \| tuple[int, int]` | `20` | Number of interpolation grid points per dimension. |
| `**kwargs` | | | Forwarded to `ax.streamplot()`. |

---

## Class: `ContourPlot` (abstract)

Abstract contour plot that interpolates data onto a regular grid before calling `ax.contour`.

```python
ContourPlot(data, bounds, N_points=200, ax=None, cbar=False, formatter=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `MeshData` | — | Data object to visualise. |
| `bounds` | `float \| tuple` | — | Domain bounds. |
| `N_points` | `int \| tuple[int, int]` | `200` | Number of grid points per dimension. |
| `cbar` | `Axes \| bool` | `False` | Whether to draw a colorbar. |

---

## Function: `animate`

```python
animate(times, fig, plots, post_draw=None, pbar=True, **kwargs) -> FuncAnimation
```

Creates a `matplotlib.animation.FuncAnimation` by calling `plot.plot(time)` for each time in `times`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `times` | `Sequence[float]` | — | Ordered sequence of simulation times. |
| `fig` | `Figure` | — | Matplotlib figure containing the plots. |
| `plots` | `tuple[Plot, ...]` | — | Plot objects to update at each frame. |
| `post_draw` | `Callable \| None` | `None` | Called after all plots at each frame; may return additional `Artist` objects. |
| `pbar` | `bool` | `True` | Show a progress bar. |
| `**kwargs` | | | Forwarded to `FuncAnimation`. |

**Example**

```python
import numpy as np
times = plot.data.time_range
anim = yp.animate(times, fig=fig, plots=(plot,))
anim.save("animation.mp4", fps=24)
```

---

## Function: `save_frames`

```python
save_frames(times, fig, plots, post_draw=None,
            output_dir="frames", prefix="frame_",
            dpi=None, pbar=True, **savefig_kwargs) -> list[str]
```

Saves each animation frame as an individual PNG file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `times` | `Sequence[float]` | — | Simulation times to render. |
| `fig` | `Figure` | — | Matplotlib figure. |
| `plots` | `tuple` | — | Plot objects to update. |
| `output_dir` | `str` | `"frames"` | Output directory (created if it does not exist). |
| `prefix` | `str` | `"frame_"` | Filename prefix; files are named `{prefix}{i:04d}.png`. |
| `dpi` | `int \| None` | `None` | Image resolution. |
| `**savefig_kwargs` | | | Extra keyword arguments forwarded to `fig.savefig()`. |

**Returns** – list of paths to the saved files.

**Example**

```python
frames = yp.save_frames(times, fig, [plot], output_dir="rho_xy", dpi=300)
```

---

## Function: `interpolate_octree_to_grid`

```python
interpolate_octree_to_grid(octree_xyz, octree_data, grid_xyz, method='linear') -> np.ndarray
```

Resamples data defined on an octree mesh onto a regular Cartesian grid using `scipy.interpolate.griddata`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `octree_xyz` | `tuple[np.ndarray, np.ndarray]` | — | Meshblock coordinate arrays. |
| `octree_data` | `np.ndarray` | — | Data on the octree mesh, shape `(N_blocks, nx, ny)`. |
| `grid_xyz` | `tuple[np.ndarray, np.ndarray]` | — | 1-D arrays defining the target regular grid. |
| `method` | `str` | `'linear'` | Interpolation method: `'linear'`, `'nearest'`, or `'cubic'`. |

**Returns** – 2-D array of shape `(len(x_grid), len(y_grid))`.

---

## Function: `make_cax`

```python
make_cax(ax) -> Axes
```

Creates a thin colorbar axes attached to the right side of `ax` using `mpl_toolkits.axes_grid1`.

---

## See Also

- [`Simulation.plot2d`](simulation.md#plot2d) / [`Simulation.animate2d`](simulation.md#animate2d) – high-level wrappers
- [`datatypes`](datatypes.md) – `Native`, `Derived`, `Vector` data loaders
- [`plot_formatter`](plot_formatter.md) – label and unit conversion
- [`decorations`](decorations.md) – default colormap and normalization settings
