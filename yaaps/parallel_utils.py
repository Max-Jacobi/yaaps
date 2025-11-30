"""
Parallel processing utilities for yaaps.

This module provides helper functions for parallel execution of tasks
using Python's multiprocessing Pool, with optional progress bars via tqdm.
"""

from multiprocessing import Pool
from sys import stdout
from typing import Callable, Iterable, Sized, Any
from functools import partial
from tqdm.auto import tqdm


################################################################################
# parallele helpers

def _apply_tail(func: Callable[..., Any], tail: tuple[Any, ...], item: Any):
    """
    Apply a function to an item with additional trailing arguments.

    This is a helper function for partial application in parallel processing.

    Args:
        func: The function to call.
        tail: Additional arguments to pass after the item.
        item: The primary argument to pass to the function.

    Returns:
        The result of func(item, *tail).
    """
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
    """
    Execute a function on items in parallel using a process pool.

    Args:
        func: The function to apply to each item. Should accept the item
            as first argument, followed by any additional args.
        itr: Iterable of items to process.
        n_cpu: Number of CPU cores to use. If 1, runs sequentially without
            spawning worker processes.
        args: Additional arguments to pass to func after each item.
        verbose: If True, display a progress bar.
        ordered: If True and n_cpu > 1, preserve the order of results
            (uses imap instead of imap_unordered).
        **kwargs: Additional keyword arguments passed to tqdm for progress bar
            customization.

    Yields:
        Results from applying func to each item in the iterable.

    Example:
        >>> def process(x, multiplier):
        ...     return x * multiplier
        >>> list(do_parallel(process, [1, 2, 3], n_cpu=2, args=(10,)))
        [10, 20, 30]  # Order may vary unless ordered=True
    """
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
    """
    Apply a function to an item and return the result with its original index.

    This helper function is used by do_parallel_enumerate to maintain
    index information through parallel processing.

    Args:
        ix_x: A tuple of (index, item) from enumerate.
        func: The function to apply to the item.
        tail: Additional arguments to pass to func after the item.

    Returns:
        A tuple of (index, result) where result is func(item, *tail).
    """
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
    """
    Execute a function on items in parallel, returning results with indices.

    Similar to do_parallel, but yields (index, result) tuples so that
    results can be matched back to their original positions even when
    processed out of order.

    Args:
        func: The function to apply to each item. Should accept the item
            as first argument, followed by any additional args.
        itr: Iterable of items to process.
        n_cpu: Number of CPU cores to use. If 1, runs sequentially.
        args: Additional arguments to pass to func after each item.
        verbose: If True, display a progress bar.
        **kwargs: Additional keyword arguments passed to tqdm.

    Yields:
        Tuples of (original_index, result) from applying func to each item.

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>> dict(do_parallel_enumerate(square, [1, 2, 3], n_cpu=2))
        {0: 1, 1: 4, 2: 9}
    """
    return do_parallel(
        _index_then_apply,
        enumerate(itr),
        n_cpu=n_cpu,
        args=(func, args),
        verbose=verbose,
        **kwargs,
    )
