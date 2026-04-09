# Units

The `units` module provides unit conversion factors and LaTeX field labels for converting simulation quantities from GRAthena++ code units to physical (CGS or natural) units. All conversions are calibrated to typical neutron star merger simulation scales (1 solar mass, geometric units).

## Import

```python
from yaaps.units import UnitConverter, FieldLabels
```

---

## Class: `UnitConverter`

Handles conversion from code units to physical units.

```python
UnitConverter()
```

Initialised with a built-in dictionary of conversion factors. Supports exact suffix matching (string keys) and regex pattern matching.

### Method: `get_conversion`

```python
converter.get_conversion(field_name) -> tuple[float, str]
```

Look up the conversion factor and unit string for a variable.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_name` | `str` | Variable name, e.g., `"rho"`, `"x1v"`. |

**Returns** – `(scale_factor, unit_string)`. Returns `(1.0, "")` if no match is found.

**Example**

```python
converter = UnitConverter()
scale, unit = converter.get_conversion("rho")
# scale ≈ 6.176e17  (code → g cm⁻³)
# unit  = " [g cm$^{-3}$]"

physical_rho = code_rho * scale
```

### Method: `add_unit`

```python
converter.add_unit(field_name, scale, unit)
```

Add or update a conversion entry.

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_name` | `str` | Variable name or regex pattern string to match. |
| `scale` | `float` | Multiplicative factor from code units to physical units. |
| `unit` | `str` | Unit string (LaTeX formatting accepted). |

**Example**

```python
converter.add_unit("my_var", 1.0e10, r" [custom unit]")
```

---

## Built-in Conversions

| Variable / Pattern | Physical Unit | Scale Factor |
|--------------------|--------------|-------------|
| `rho` | g cm⁻³ | 6.1758 × 10¹⁷ |
| `aux.e` | g cm⁻³ | 6.1758 × 10¹⁷ |
| `aux.T` | MeV | 1.0 |
| `aux.s` | k_B | 1.0 |
| `eps` | erg g⁻¹ | 8.9876 × 10²⁰ |
| `P` | erg cm⁻³ | 5.5507 × 10³⁸ |
| `energy` | erg | 1.7871 × 10⁵³ |
| `time` | ms | 4.9255 × 10⁻³ |
| `r`, `x`, `y`, `z` | km | 1.4766 |
| `mass` | M☉ | 1.0 |
| `Omega` | s⁻¹ | 2.0303 × 10⁵ |
| `x1v`, `x2v`, `x3v`, `x1f`, `x2f`, `x3f` | km | 1.4766 |
| `nu{N}_lum` (regex) | erg s⁻¹ | 3.6281 × 10⁵⁹ |
| `nu{N}_en` (regex) | MeV | 1.1155 × 10⁶⁰ |
| `m_ej` (regex) | M☉ | 1.0 |
| `mdot_ej` (regex) | M☉ s⁻¹ | 2.0303 × 10⁵ |
| `util_u` (regex) | c | 1.0 |
| `vel` (regex) | c | 1.0 |
| `x[1-3][vf]?` (regex) | km | 1.4766 |

---

## Class: `FieldLabels`

Provides LaTeX-formatted labels for field names used in publication-quality plots.

```python
FieldLabels()
```

### Method: `get_label`

```python
labels.get_label(field_name) -> str
```

Return the LaTeX label for a variable. If the name is not in the mapping, returns the name unchanged.

**Example**

```python
labels = FieldLabels()
labels.get_label("rho")       # '$\\rho$'
labels.get_label("x1v")       # '$x$'
labels.get_label("time")      # '$t$'
labels.get_label("unknown")   # 'unknown'
```

### Method: `add_label`

```python
labels.add_label(field_name, label)
```

Add or update a label entry.

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_name` | `str` | Variable name. |
| `label` | `str` | LaTeX-formatted label string. |

**Example**

```python
labels.add_label("my_var", r"$\tilde{Q}$")
```

---

## Built-in Labels

| Variable | LaTeX label |
|----------|-------------|
| `x1`, `x1v`, `x1f` | `$x$` |
| `x2`, `x2v`, `x2f` | `$y$` |
| `x3`, `x3v`, `x3f` | `$z$` |
| `time` | `$t$` |
| `rho` / `hydro.prim.rho` | `$\rho$` |
| `p` / `P` / `hydro.prim.p` | `$P$` |
| `eps` | `$\varepsilon$` |
| `hydro.aux.s` | `$s$` |
| `hydro.aux.T` | `$T$` |
| `hydro.aux.e` | `$e$` |
| `ye` / `passive_scalar.r_0` | `$Y_e$` |
| `util_x` / `hydro.prim.util_u_1` | `$\tilde{u}^x$` |
| `util_y` / `hydro.prim.util_u_2` | `$\tilde{u}^y$` |
| `util_z` / `hydro.prim.util_u_3` | `$\tilde{u}^z$` |
| `B_x` / `B.Bcc_1` | `$B^x$` |
| `B_y` / `B.Bcc_2` | `$B^y$` |
| `B_z` / `B.Bcc_3` | `$B^z$` |
| `b_x` / `field.aux.b_u_1` | `$b^x$` |
| `b_y` / `field.aux.b_u_2` | `$b^y$` |
| `b_z` / `field.aux.b_u_3` | `$b^z$` |

---

## See Also

- [`plot_formatter`](plot_formatter.md) – uses `UnitConverter` and `FieldLabels` for label formatting
- [`decorations`](decorations.md) – default colormap and normalization per variable
