# Input

The `Input` class reads and parses Athena++ parameter files (`.inp` or `.par`). It exposes the parsed parameters through a dictionary-like interface with slash-separated key notation.

## Import

```python
from yaaps.input import Input
```

---

## Class: `Input`

```python
Input(file_path)
```

Reads the parameter file at `file_path` and stores all sections and key-value pairs.

**File format** – sections are delimited by `<section_name>` headers, followed by `key = value` pairs. Comments starting with `#` or `//` are stripped. The file is read up to a `<par_end>` line or until the end of the file.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the Athena++ input or restart file. |

**Raises**

- `ValueError` – if the file does not start with a section header or contains an invalid line.
- `RuntimeError` – if the file exceeds the maximum supported length (`10 000` lines).

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `dict` | Nested dictionary `{section: {key: value}}`. |
| `git_hash` | `str` | Git hash string extracted from the file header, if present. |

---

## Dictionary Interface

`Input` implements Python's `Mapping` protocol.

### `__getitem__`

```python
inp["section/key"]
```

Returns the value for `key` inside `section`. The slash separator can be chained for deeper nesting. Raises `KeyError` if not found.

**Example**

```python
inp = Input("simulation.par")

nx1      = inp["mesh/nx1"]        # e.g. 128
problem  = inp["job/problem_id"]  # e.g. "bns"
x1max    = inp["mesh/x1max"]      # e.g. 1024.0
```

### `keys`

```python
inp.keys()
```

Returns the top-level section names.

### `__contains__`

```python
"mesh/nx1" in inp  # True / False
```

Supports membership testing using slash-separated keys.

---

## Value Interpretation

Values are automatically converted to the most specific Python type:

| Raw string | Converted type |
|-----------|---------------|
| `"128"` | `int` |
| `"1.5e3"` | `float` |
| `"true"` / `"false"` | `bool` |
| `"[1, 2, 3]"` | `list` |
| anything else | `str` |

Multi-line list values (opening `[` without closing `]`) are accumulated over subsequent lines.

---

## Method: `diff`

```python
inp.diff(other, float_tol=1e-8) -> dict[str, dict[str, Any]]
```

Compares two `Input` instances and returns all differing parameters.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `Input` | — | The second `Input` to compare against. |
| `float_tol` | `float` | `1e-8` | Relative tolerance for floating-point comparisons: two floats are considered equal when `2*(v1-v2) < float_tol*(v1+v2)`. |

**Returns** – a nested dictionary `{section: {key: (value_in_self, value_in_other)}}`. Missing values are represented as `" - "`.

**Example**

```python
inp1 = Input("run_A.par")
inp2 = Input("run_B.par")

diff = inp1.diff(inp2)
for section, params in diff.items():
    for key, (v1, v2) in params.items():
        print(f"{section}/{key}: {v1}  vs  {v2}")
```

---

## Command-Line Utility

The `examples/diff_inputs.py` script wraps `Input.diff` for convenient CLI use:

```bash
python examples/diff_inputs.py run_A.par run_B.par
python examples/diff_inputs.py run_A.par run_B.par --float_tol 1e-6
python examples/diff_inputs.py run_A.par run_B.par --ignore output surface
```

---

## See Also

- [`Simulation`](simulation.md) – uses `Input` internally to read simulation parameters
