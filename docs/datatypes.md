# Data Types

The `datatypes` module defines abstract and concrete classes for loading and interpolating simulation data from GRAthena++ `.athdf` output files. It supports native (directly stored) variables, derived (computed) quantities, and vector fields.

## Import

```python
from yaaps.datatypes import MeshData, Native, Derived, Vector
```

---

## Type Alias: `Sampling`

```python
Sampling = tuple[str, str] | str
```

Specifies the coordinate sampling. Can be either:

- A tuple such as `('x1v', 'x2v')` — uses the exact coordinate names.
- A shorthand string such as `'xy'`, `'xz'`, or `'yz'` — automatically converted to the equivalent tuple.

**Shorthand mapping**

| Key | Coordinate name |
|-----|----------------|
| `'x'` | `'x1v'` |
| `'y'` | `'x2v'` |
| `'z'` | `'x3v'` |

---

## Class: `MeshData` (abstract)

Abstract base class that defines the shared interface for all mesh-based data loaders.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sim` | `Simulation` | Reference to the parent [`Simulation`](simulation.md) object. |
| `var` | `str` | Variable name. |
| `sampling` | `tuple[str, str]` | Coordinate sampling names, e.g., `('x1v', 'x2v')`. |
| `iter_range` | `np.ndarray` | Array of available iteration numbers. |
| `time_range` | `np.ndarray` | Array of available simulation times. |

### Method: `load_data` (abstract)

```python
load_data(time, strip_ghosts=True) -> tuple
```

Load data at the given time. Implemented by each concrete subclass.

**Returns** – `(xyz, data, actual_time)` where:
- `xyz` – tuple of coordinate arrays `(x_per_block, y_per_block)`.
- `data` – NumPy array of variable values.
- `actual_time` – the actual simulation time of the loaded snapshot.

---

### Method: `interp`

```python
interp(points, time, method='linear') -> np.ndarray
```

Interpolate data at arbitrary spatial points and a given time.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `np.ndarray` | — | Array of shape `(N, 2)` or `(..., 2)` giving the (x, y) query points. |
| `time` | `float` | — | Simulation time to load data at. |
| `method` | `str` | `'linear'` | Interpolation method: `'nearest'`, `'linear'`, `'slinear'`, `'cubic'`, `'quintic'`, or `'pchip'`. |

**Returns** – Array of interpolated values with shape matching the leading dimensions of `points`. Points outside the domain are set to `NaN`.

---

## Class: `Native`

Loads variable data directly from simulation output files.

```python
Native(sim, var, sampling=('x1v', 'x2v'))
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | `Simulation` | — | Parent simulation object. |
| `var` | `str` | — | Variable name or alias (e.g., `"rho"` is resolved to `"hydro.prim.rho"`). |
| `sampling` | `Sampling` | `('x1v', 'x2v')` | Coordinate sampling specification. |

### Method: `load_data`

```python
load_data(time, strip_ghosts=True) -> tuple
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | `float` | — | Simulation time; the nearest available snapshot is loaded. |
| `strip_ghosts` | `bool` | `True` | If `True`, ghost zone cells are removed from the returned data. |

**Returns** – `(xyz, data, actual_time)`.

**Example**

```python
from yaaps import Simulation
from yaaps.datatypes import Native

sim = Simulation("/path/to/simulation")
rho = Native(sim, "rho")
xyz, data, time = rho.load_data(100.0)
```

---

## Class: `Derived`

Computes derived quantities from one or more native variables using a user-supplied function.

```python
Derived(sim, var, depends, definition, sampling=('x1v', 'x2v'))
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | `Simulation` | — | Parent simulation object. |
| `var` | `str` | — | Name for the derived variable. |
| `depends` | `tuple[str, ...]` | — | Variable names that the derived quantity depends on. |
| `definition` | `Callable` | — | Function to compute the derived quantity. Receives one array per dependency as positional arguments. May optionally accept `xyz`, `time`, and `sampling` keyword arguments. |
| `sampling` | `Sampling` | `('x1v', 'x2v')` | Coordinate sampling specification. |

`iter_range` and `time_range` are automatically computed as the intersection of the available times of all dependencies.

### Method: `load_data`

```python
load_data(time, strip_ghosts=True) -> tuple
```

Loads all dependencies and applies `definition` to produce the derived data.

**Raises**

- `RuntimeError` – if dependent variables have inconsistent time grids.

**Example**

```python
from yaaps.datatypes import Derived
import numpy as np

def velocity_magnitude(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)

vmag = Derived(sim, "|v|",
               depends=("util_x", "util_y", "util_z"),
               definition=velocity_magnitude)
xyz, data, time = vmag.load_data(100.0)
```

---

## Class: `Vector`

Combines two or three scalar `MeshData` objects into a 2-D vector field.

```python
Vector(components)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `components` | `tuple[MeshData, MeshData, MeshData]` | Three `MeshData` objects for the x, y, and z components. The pair matching the current sampling is selected automatically. |

**Raises**

- `RuntimeError` – if components belong to different simulations or have different sampling.

### Class Method: `from_native`

```python
Vector.from_native(sim, vars, sampling=('x1v', 'x2v')) -> Vector
```

Convenience constructor that creates `Native` objects internally.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim` | `Simulation` | — | Parent simulation object. |
| `vars` | `tuple[str, str, str]` | — | Variable names for the x, y, z components. |
| `sampling` | `Sampling` | `('x1v', 'x2v')` | Coordinate sampling specification. |

**Example**

```python
from yaaps.datatypes import Vector

vel = Vector.from_native(sim, ("util_x", "util_y", "util_z"))
```

### Method: `load_data`

```python
load_data(time) -> tuple
```

**Returns** – `(xyz, data, actual_time)` where `data` has shape `(2, N_blocks, nx, ny)` with the two selected vector components stacked along the first axis.

### Method: `interp`

```python
interp(points, time, method='linear') -> np.ndarray
```

Interpolates both vector components and returns an array of shape `(2, ...)`.

---

## See Also

- [`Simulation`](simulation.md)
- [`plot2D`](plot2D.md) – plot classes that consume `MeshData` objects
