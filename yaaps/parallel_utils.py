from multiprocessing import Pool
from sys import stdout
from typing import Callable, Iterable, Sized, Any
from functools import partial
from tqdm import tqdm


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

    func_args = partial(_apply_tail, func, args)

    if n_cpu == 1:
        return tqdm(map(func_args, itr), **kwargs)
    pool = Pool(n_cpu)
    if ordered:
        return tqdm(pool.imap(func_args, itr), **kwargs)
    return tqdm(pool.imap_unordered(func_args, itr), **kwargs)

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
