# Simulation

The `Simulation` class is the main entry point for accessing GRAthena++ simulation data. It provides a unified interface for loading simulation parameters, accessing time-series data, waveform extractions, tracer particles, horizon data, and creating 2D visualisations.

## Import

```python
from yaaps import Simulation
```

---

## Constructor

```python
Simulation(path, input_path=None)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to the simulation output directory. |
| `input_path` | `str \| None` | `None` | Explicit path to the `.inp` or `.par` parameter file. If `None`, the constructor searches `path` for a valid file automatically. |

**Raises**

- `FileNotFoundError` – if no valid parameter file is found in the directory.

**Example**

```python
sim = Simulation("/path/to/simulation/output")
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Absolute path to the simulation directory. |
| `name` | `str` | Short name derived from the directory path. |
| `input` | `Input` | Parsed [`Input`](input.md) object containing all simulation parameters. |
| `problem_id` | `str` | The problem ID from the input file (`job/problem_id`). |
| `dx` | `list[float]` | Grid spacings `[dx1, dx2, dx3]` at the finest refinement level. |
| `md` | `dict` | Mutable metadata dictionary; may contain `"t_merg"` for merger offset. If `metadata.json` exists in the simulation directory it is read into this dict |

---

## Properties

### `hst`

```python
sim.hst  # -> dict
```

Loads and returns the history (`.hst`) file data. The result is cached after the first call.

**Returns** – `dict` mapping column names to 1-D NumPy arrays, sorted and deduplicated by iteration or time.

**Example**

```python
times = sim.hst["time"]
max_rho = sim.hst["max_rho"]
```

---

### `scrape`

```python
sim.scrape  # -> scrape_dir_athdf
```

Returns the directory scraper (`scrape_dir_athdf`) used internally to locate and read `.athdf` output files. Cached after the first access.

---

## Methods

### `wav`

```python
sim.wav(radius, prefix="wav") -> dict
```

Loads waveform extraction data at the given radius.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `float` | — | Extraction radius in code units. |
| `prefix` | `str` | `"wav"` | Filename prefix; the file is expected at `{path}/{prefix}_r{radius:.2f}.txt`. |

**Returns** – `dict` mapping column names to NumPy arrays.

---

### `tra`

```python
sim.tra(index, prefix="tra") -> dict
```

Loads tracker (e.g., extrema or puncture) data for the given tracker index.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index` | `int` | — | Tracker index. |
| `prefix` | `str` | `"tra"` | Filename prefix; the file is expected at `{path}/{prefix}.ext{index}.txt`. |

**Returns** – `dict` mapping column names to NumPy arrays.

---

### `horizon`

```python
sim.horizon(index) -> dict
```

Loads apparent horizon summary data for the given horizon index.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `int` | Horizon index (typically `0` or `1` for binary systems). |

**Returns** – `dict` mapping column names to NumPy arrays.

---

### `available`

```python
sim.available(var) -> list
```

Returns a list of `(sampling, ghosts)` tuples describing the sampling configurations and ghost zone settings that are available for the given variable.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `str` | Variable name to query. |

---

### `complete_var`

```python
sim.complete_var(var, sampling) -> tuple[str, bool]
```

Resolves an abbreviated variable name to its full internal name and determines whether ghost zones are present.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `str` | Short or full variable name (e.g., `"rho"` → `"hydro.prim.rho"`). |
| `sampling` | `tuple[str, str]` | Coordinate sampling directions, e.g., `('x1v', 'x2v')`. |

**Returns** – `(full_variable_name, ghost_zones)`.

**Raises**

- `ValueError` – if the variable cannot be resolved uniquely or the requested sampling is not available.

---

### `plot2d`

```python
sim.plot2d(time, *args, **kwargs) -> NativeColorPlot
```

Creates a 2D colour plot of simulation data at the given time.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | `float` | Simulation time to plot. |
| `var` | `str` | Variable name to plot (keyword argument). |
| `sampling` | `str \| tuple` | Coordinate sampling, e.g., `'xy'` or `('x1v', 'x2v')`. |
| `cmap` | `str` | Matplotlib colormap name. |
| `norm` | `str` | Normalization: `'log'`, `'lin'`, or `'asinh'`. |
| `vmin`, `vmax` | `float` | Colour scale limits. |
| `draw_meshblocks` | `bool` | Overlay mesh-block boundaries. |
| `formatter` | `str \| PlotFormatterBase` | `"raw"` or `"paper"` (or a formatter object). |

**Returns** – a [`NativeColorPlot`](plot2D.md#nativecolorplot) object.

**Example**

```python
sim.plot2d(time=100.0, var="rho", norm="log", sampling="xy")
```

---

### `animate2d`

```python
sim.animate2d(times, *args, **kwargs) -> FuncAnimation
```

Creates a matplotlib `FuncAnimation` of 2D simulation data over a sequence of times. Accepts the same keyword arguments as `plot2d`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `times` | `Iterable[float]` | Simulation times to include in the animation. |

**Returns** – a `matplotlib.animation.FuncAnimation` object.

**Example**

```python
import numpy as np
times = np.linspace(0, 200, 50)
anim = sim.animate2d(times, var="rho", norm="log")
anim.save("rho.mp4")
```

---

## See Also

- [`Input`](input.md) – parameter file parsing
- [`plot2D`](plot2D.md) – all plot classes
- [`datatypes`](datatypes.md) – `Native`, `Derived`, `Vector` data loaders
