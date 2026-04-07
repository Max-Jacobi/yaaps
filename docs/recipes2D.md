# 2D Recipes (`recipes2D`)

The `recipes2D` module provides mathematical utility functions for working with 3x3 metric tensors and coordinate transformations. These are primarily used when defining derived quantities in GR hydrodynamics simulations.

## Import

```python
from yaaps.recipes2D import det, gup, raise_lower, untangle_xyz, radial_proj, normalize_vec, absolute_val
```

All public names are also re-exported by `plot2D` via `from .recipes2D import *`.

---

## Function: `det`

```python
det(gxx, gxy, gxz, gyy, gyz, gzz) -> array-like
```

Compute the determinant of a 3x3 symmetric metric tensor.

**Parameters** – the six independent components of the covariant metric tensor (`gxx`, `gxy`, `gxz`, `gyy`, `gyz`, `gzz`). Each may be a scalar or NumPy array.

**Returns** – the determinant `g = det(g_ij)`.

**Example**

```python
from yaaps.recipes2D import det
g = det(gxx, gxy, gxz, gyy, gyz, gzz)
```

---

## Function: `gup`

```python
gup(gxx, gxy, gxz, gyy, gyz, gzz) -> tuple
```

Compute the contravariant (inverse) components of a 3x3 symmetric metric tensor.

**Returns** – tuple `(guxx, guxy, guxz, guyy, guyz, guzz)` where `g^{ij} = cofactor(g_{ij}) / det(g)`.

---

## Function: `raise_lower`

```python
raise_lower(vx, vy, vz, gxx, gxy, gxz, gyy, gyz, gzz) -> tuple
```

Contract a vector with a metric tensor to convert between covariant and contravariant components.

- **Lowering**: pass the covariant metric `g_ij` to obtain `v_i = g_ij v^j`.
- **Raising**: pass the contravariant metric `g^{ij}` to obtain `v^i = g^{ij} v_j`.

**Returns** – tuple `(vtx, vty, vtz)` of transformed vector components.

**Example**

```python
from yaaps.recipes2D import raise_lower, gup

# Raise a covariant vector
vu_x, vu_y, vu_z = raise_lower(vd_x, vd_y, vd_z, *gup(gxx, gxy, gxz, gyy, gyz, gzz))
```

---

## Function: `untangle_xyz`

```python
untangle_xyz(xyz, samp) -> np.ndarray
```

Convert mesh-block coordinate data into a uniform 3D NumPy array.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `xyz` | `tuple` | Coordinate arrays organised by mesh blocks (as returned by `MeshData.load_data`). |
| `samp` | `tuple[str, str]` | Sampling specification, e.g., `('x1v', 'x2v')`, used to determine spatial indices. |

**Returns** – array of shape `(3, N_blocks, nx1, nx2)` containing meshgrid coordinates for all three spatial directions.

---

## Function: `radial_proj`

```python
radial_proj(vdx, vdy, vdz, *gd, xyz, sampling) -> array-like
```

Compute the radial projection of a covariant vector: `(x^i v_i) / r`, where `r` is the proper radial distance computed from the metric.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vdx, vdy, vdz` | array-like | Covariant vector components. |
| `*gd` | array-like | Six independent components of the covariant metric tensor `(gxx, gxy, gxz, gyy, gyz, gzz)`. |
| `xyz` | keyword | Mesh-block coordinate arrays. |
| `sampling` | keyword | Sampling specification tuple, e.g., `('x1v', 'x2v')`. |

**Returns** – radial projection scalar field.

**Example**

```python
# Used as a definition for a Derived quantity:
from yaaps.recipes2D import radial_proj

proj = Derived(sim, "v_r",
               depends=("vel_x", "vel_y", "vel_z",
                        "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"),
               definition=radial_proj)
```

---

## Function: `normalize_vec`

```python
normalize_vec(vx, vy, vz, *gd) -> tuple
```

Normalize a vector to unit length using the metric tensor. Computes the norm `|v| = sqrt(v^i g_{ij} v^j)` and divides each component by it.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vx, vy, vz` | array-like | Vector components (covariant or contravariant; the metric must match). |
| `*gd` | array-like | Six independent metric components used to contract the vector. |

**Returns** – tuple `(nvx, nvy, nvz)` of normalized components.

---

## Function: `absolute_val`

```python
absolute_val(vx, vy, vz, *gd) -> array-like
```

Compute the norm (absolute value) of a vector: `sqrt(v^i g_{ij} v^j)`.

**Parameters** – same as `normalize_vec`.

**Returns** – scalar norm field.

**Example**

```python
from yaaps.recipes2D import absolute_val

vmag = Derived(sim, "|v|",
               depends=("vel_x", "vel_y", "vel_z",
                        "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"),
               definition=absolute_val)
```

---

## See Also

- [`datatypes`](datatypes.md) – `Derived` class that uses these functions as `definition` callbacks
- [`plot2D`](plot2D.md) – re-exports all public names from this module
