from multiprocessing import Pool
from sys import stdout
from typing import Callable, Iterable, Sized, Any
from functools import partial
from tqdm.auto import tqdm


################################################################################
# parallele helpers

def _apply_tail(func: Callable[..., Any], tail: tuple[Any, ...], item: Any):
    return func(item, *tail)

def do_parallel[R](
    func: Callable[..., R],
    itr: Iterable,
    n_cpu: int,
    args: tuple[Any, ...] = (),
    verbose: bool = False,
    ordered: bool = False,
    **kwargs
) -> Iterable[R]:
    if isinstance(itr, Sized):
        kwargs["total"] = len(itr)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)
    kwargs.setdefault("position", 0)
    kwargs.setdefault("ascii", True)

    func_args = partial(_apply_tail, func, args)

    if n_cpu == 1:
        yield from tqdm(map(func_args, itr), **kwargs)
        return
    with Pool(n_cpu) as pool:
        if ordered:
            yield from tqdm(pool.imap(func_args, itr), **kwargs)
        yield from tqdm(pool.imap_unordered(func_args, itr), **kwargs)

def _index_then_apply[T, R](ix_x: tuple[int, T], func: Callable[..., R], tail: tuple[Any, ...]) -> tuple[int, R]:
    i, x = ix_x
    return i, func(x, *tail)

def do_parallel_enumerate[T, R](
    func: Callable[..., R],
    itr: Iterable[T],
    n_cpu: int,
    args: tuple[Any, ...] = (),
    verbose: bool = False,
    **kwargs
) -> Iterable[tuple[int, R]]:
    return do_parallel(
        _index_then_apply,
        enumerate(itr),
        n_cpu=n_cpu,
        args=(func, args),
        verbose=verbose,
        **kwargs,
    )
