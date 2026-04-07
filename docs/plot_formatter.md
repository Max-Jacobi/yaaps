# Plot Formatter

The `plot_formatter` module controls how axis labels, titles, colorbar labels, and data values are formatted and converted between code units and physical units in all plot classes.

## Import

```python
from yaaps.plot_formatter import PlotFormatter, RawPlotFormatter, PaperPlotFormatter
```

---

## Type Alias: `PlotMode`

```python
PlotMode = Literal["raw", "paper"]
```

| Mode | Description |
|------|-------------|
| `"raw"` | Code-internal variable names; no unit conversion. Suitable for quick inspection. |
| `"paper"` | LaTeX labels with physical units; suitable for publication-quality figures. |

---

## Factory Function: `PlotFormatter`

```python
PlotFormatter(mode="raw", unit_converter=None, field_labels=None) -> PlotFormatterBase
```

Returns the appropriate concrete formatter for the given mode. This is the recommended way to create a formatter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `PlotMode` | `"raw"` | `"raw"` returns a `RawPlotFormatter`; `"paper"` returns a `PaperPlotFormatter`. |
| `unit_converter` | `UnitConverter \| None` | `None` | Custom converter; defaults to `UnitConverter()`. |
| `field_labels` | `FieldLabels \| None` | `None` | Custom label map; defaults to `FieldLabels()`. |

**Example**

```python
from yaaps.plot_formatter import PlotFormatter

raw    = PlotFormatter("raw")
paper  = PlotFormatter("paper")
```

---

## Class: `PlotFormatterBase` (abstract)

Abstract base class defining the interface for all formatters.

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `mode` | `PlotMode` | `"raw"` or `"paper"`. |
| `unit_converter` | `UnitConverter` | Instance used for unit look-ups. |
| `field_labels` | `FieldLabels` | Instance used for LaTeX label look-ups. |

### Abstract Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `format_axis_label` | `(field_name) -> str` | Returns a formatted axis label. |
| `format_title` | `(field_name, time) -> str` | Returns a formatted plot title with time. |
| `format_colorbar_label` | `(field_name) -> str` | Returns a formatted colorbar label. |
| `convert_data` | `(field_name, data) -> np.ndarray` | Converts data from code units to display units. |
| `convert_coordinate` | `(coord_name, coord_data) -> np.ndarray` | Converts coordinates from code units to display units. |
| `inverse_convert_time` | `(time) -> float` | Converts time from display units back to code units. |
| `inverse_convert_coordinate` | `(coord_name, coord_data) -> np.ndarray` | Converts coordinates from display units back to code units. |

---

## Class: `RawPlotFormatter`

Formatter for raw mode. Returns all names unchanged and performs no unit conversions (identity transformations).

```python
RawPlotFormatter(unit_converter=None, field_labels=None)
```

**Behaviour**

| Method | Returns |
|--------|---------|
| `format_axis_label("rho")` | `'rho'` |
| `format_title("rho", 100.0)` | `'rho @ t= 100.00'` |
| `format_colorbar_label("rho")` | `'rho'` |
| `convert_data("rho", data)` | `data` (unchanged) |
| `convert_coordinate("x1v", c)` | `c` (unchanged) |
| `inverse_convert_time(t)` | `t` (unchanged) |

---

## Class: `PaperPlotFormatter`

Formatter for paper mode. Returns LaTeX labels with physical unit strings and converts data/coordinates using `UnitConverter`.

```python
PaperPlotFormatter(unit_converter=None, field_labels=None)
```

**Behaviour**

| Method | Example return value |
|--------|---------------------|
| `format_axis_label("rho")` | `'$\rho$ [g cm$^{-3}$]'` |
| `format_axis_label("x1v")` | `'$x$ [km]'` |
| `format_title("rho", 100.0)` | `'$\rho$ @ $t$ = 0.49 ms'` |
| `format_colorbar_label("rho")` | `'$\rho$ [g cm$^{-3}$]'` |
| `convert_data("rho", data)` | `data * 6.176e17` (code → g cm⁻³) |
| `convert_coordinate("x1v", c)` | `c * 1.4766` (code → km) |
| `inverse_convert_time(0.49)` | `≈ 100.0` (ms → code) |

---

## Using a Formatter With Plots

All plot classes accept a `formatter` argument:

```python
import yaaps as ya
import yaaps.plot2D as yp
from yaaps.plot_formatter import PlotFormatter

sim = ya.Simulation("/path/to/sim")

formatter = PlotFormatter("paper")
plot = yp.NativeColorPlot(sim, var="rho", formatter=formatter)
plot.plot(100.0)
```

You can also pass the string shorthand directly:

```python
plot = yp.NativeColorPlot(sim, var="rho", formatter="paper")
```

### Switching Mode After Construction

```python
plot.set_plot_mode("paper")
plot.set_plot_mode("raw")
```

### Converting Times Between Units

When accepting user-supplied times in physical units (e.g., milliseconds), convert them to code units before passing to `plot`:

```python
formatter = PlotFormatter("paper")
time_in_ms = 4.925  # milliseconds
time_code  = formatter.inverse_convert_time(time_in_ms)
plot.plot(time_code)
```

---

## See Also

- [`units`](units.md) – `UnitConverter` and `FieldLabels` used internally
- [`plot2D`](plot2D.md) – plot classes that accept a formatter
