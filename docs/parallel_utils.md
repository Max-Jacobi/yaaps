# Parallel Utilities (`parallel_utils`)

The `parallel_utils` module provides helpers for executing tasks in parallel using Python's `multiprocessing.Pool`, with optional `tqdm` progress bars.

## Import

```python
from yaaps.parallel_utils import do_parallel, do_parallel_enumerate
```

---

## Function: `do_parallel`

```python
do_parallel(func, itr, n_cpu, args=(), verbose=False, ordered=False, **kwargs) -> Iterable[R]
```

Apply `func` to each item in `itr` using `n_cpu` worker processes. If `n_cpu == 1`, runs sequentially without spawning worker processes.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | — | Function to apply. Called as `func(item, *args)`. |
| `itr` | `Iterable` | — | Items to process. |
| `n_cpu` | `int` | — | Number of worker processes. `1` runs sequentially. |
| `args` | `tuple` | `()` | Extra positional arguments appended to each `func` call. |
| `verbose` | `bool` | `False` | If `True`, display a `tqdm` progress bar. |
| `ordered` | `bool` | `False` | If `True` and `n_cpu > 1`, preserve input order (uses `Pool.imap`). Otherwise uses `Pool.imap_unordered` for better throughput. |
| `**kwargs` | | | Extra keyword arguments forwarded to `tqdm`. |

**Yields** – results from applying `func` to each item.

**Example**

```python
from yaaps.parallel_utils import do_parallel

def process(time, sim):
    return sim.hst["rho"][sim.hst["time"] == time]

results = list(do_parallel(process, times, n_cpu=4, args=(sim,), verbose=True))
```

---

## Function: `do_parallel_enumerate`

```python
do_parallel_enumerate(func, itr, n_cpu, args=(), verbose=False, **kwargs) -> Iterable[tuple[int, R]]
```

Same as `do_parallel`, but yields `(original_index, result)` tuples so that results can be matched back to their position in `itr` even when processed out of order.

**Parameters** – identical to `do_parallel` (no `ordered` argument; results always carry their index).

**Yields** – tuples `(i, result)` where `i` is the original 0-based index of the item.

**Example**

```python
from yaaps.parallel_utils import do_parallel_enumerate

def square(x):
    return x ** 2

indexed = dict(do_parallel_enumerate(square, [10, 20, 30], n_cpu=2))
# indexed == {0: 100, 1: 400, 2: 900}
```

---

## Notes

- When `n_cpu == 1`, both functions fall back to a sequential `map` call without creating a `Pool`. This is useful for debugging or when the overhead of inter-process communication would exceed the computation time.
- Progress bars are provided by `tqdm.auto`, which automatically selects the notebook or terminal variant.
- The `tqdm` `leave` option defaults to `False` so bars are removed after completion.

---

## See Also

- [`plot2D.save_frames`](plot2D.md#function-save_frames) – uses sequential frame saving; `do_parallel` can be used to parallelise custom frame-generation pipelines.
