# Decorations

The `decorations` module configures default colormap and normalization settings for known simulation variables, and provides the variable alias dictionary used throughout the package.

## Import

```python
from yaaps.decorations import update_color_kwargs, var_alias
```

---

## Variable Aliases

The `var_alias` dictionary maps short, human-friendly variable names to their full internal names used in the HDF5 output files.

```python
from yaaps.decorations import var_alias
```

| Short alias | Full internal name |
|-------------|-------------------|
| `"rho"` | `"hydro.prim.rho"` |
| `"p"` | `"hydro.prim.p"` |
| `"ye"` | `"passive_scalar.r_0"` |
| `"s"` | `"hydro.aux.s"` |
| `"util_x"` | `"hydro.prim.util_u_1"` |
| `"util_y"` | `"hydro.prim.util_u_2"` |
| `"util_z"` | `"hydro.prim.util_u_3"` |
| `"B_x"` | `"B.Bcc_1"` |
| `"B_y"` | `"B.Bcc_2"` |
| `"B_z"` | `"B.Bcc_3"` |
| `"b_x"` | `"field.aux.b_u_1"` |
| `"b_y"` | `"field.aux.b_u_2"` |
| `"b_z"` | `"field.aux.b_u_3"` |
| `"m1_E_e"` | `"M1.lab.sc_E_00"` |
| `"m1_E_a"` | `"M1.lab.sc_E_01"` |
| `"m1_E_x"` | `"M1.lab.sc_E_02"` |
| `"m1_nG_e"` | `"M1.lab.sc_nG_00"` |
| `"m1_nG_a"` | `"M1.lab.sc_nG_01"` |
| `"m1_nG_x"` | `"M1.lab.sc_nG_02"` |
| `"m1_J_e"` | `"M1.rad.sc_J_00"` |
| `"m1_J_a"` | `"M1.rad.sc_J_01"` |
| `"m1_J_x"` | `"M1.rad.sc_J_02"` |
| `"m1_n_e"` | `"M1.rad.sc_n_00"` |
| `"m1_n_a"` | `"M1.rad.sc_n_01"` |
| `"m1_n_x"` | `"M1.rad.sc_n_02"` |

Aliases are resolved automatically by `Simulation.complete_var` whenever a `Native` data object or plot is created.

---

## Function: `update_color_kwargs`

```python
update_color_kwargs(var, kwargs, data) -> dict
```

Applies default colormap and normalization settings for the given variable name, then derives `vmin`/`vmax` from the data if not already provided. Returns an updated copy of `kwargs` ready to pass to `pcolormesh`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `str` | Full internal variable name used to look up defaults. |
| `kwargs` | `dict` | Existing keyword arguments (e.g., user-supplied `cmap`, `norm`, `vmin`, `vmax`). User-supplied values always take precedence over defaults. |
| `data` | `np.ndarray` | Data array used to compute `vmin`/`vmax` when they are absent. |

**Returns** – updated `dict` with the `'norm'` key replaced by a concrete matplotlib
`Normalize`, `LogNorm`, or `AsinhNorm` instance.

**Raises**

- `ValueError` – if an unrecognised normalization string is given. Valid values: `'log'`, `'lin'`, `'asinh'`.

**Normalization details**

| `norm` string | Matplotlib class | Notes |
|---------------|-----------------|-------|
| `'lin'` | `Normalize` | Linear scale. |
| `'log'` | `LogNorm` | Log scale; non-positive data is excluded from auto-range. |
| `'asinh'` | `AsinhNorm` | Hyperbolic arc-sine scale; `linear_width` defaults to `vmax * 1e-7`. |

**Example**

```python
from yaaps.decorations import update_color_kwargs
import numpy as np

data = np.random.rand(100, 100) * 1e14
kwargs = update_color_kwargs("hydro.prim.rho", {}, data)
# kwargs now contains: {'cmap': 'magma', 'norm': LogNorm(...)}
```

---

## Default Colormap / Normalization Table

| Variable | Colormap | Norm | vmin |
|----------|----------|------|------|
| `hydro.prim.rho` | `magma` | log | — |
| `passive_scalar.r_0` (Ye) | `coolwarm_r` | lin | — |
| `hydro.aux.u_t` | `RdBu` | lin | -1.1 |
| `hydro.aux.T` | `hot` | lin | — |
| `hydro.aux.e` | `plasma` | log | — |
| `hydro.aux.hu_t` | `managua` | lin | -1.1 |
| `M1.lab.sc_E_00/01/02` | `plasma` | log | 1e-14 |
| `M1.lab.sc_nG_00/01/02` | `viridis` | log | 1e45 |
| `M1.rad.sc_J_00/01/02` | `plasma` | log | 1e-14 |
| `M1.rad.sc_n_00/01/02` | `viridis` | log | 1e45 |
| `geom.con.H`, `geom.con.M`, `geom.con.C` | `cubehelix` | log | — |
| *(default)* | `viridis` | lin | — |

Variables not listed fall back to the default `viridis` / linear normalization.

---

## See Also

- [`units`](units.md) – `UnitConverter` and `FieldLabels`
- [`plot_formatter`](plot_formatter.md) – uses decorations indirectly through `update_color_kwargs`
- [`plot2D`](plot2D.md) – calls `update_color_kwargs` before every `pcolormesh` draw
