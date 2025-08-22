import numpy as np
import h5py as h5
from typing import Callable

from .surface_func import SurfaceFunc

def _parse_group(hf, grp: str) -> dict[str, str]:
    res = {}
    for key in hf[grp]:
        if isinstance(hf[f"{grp}/{key}"], h5.Dataset):
            res[key] = f"{grp}/{key}"
        else:
            res = {**res, **_parse_group(hf, f"{grp}/{key}")}
    return res

def parse_h5(
    path: str,
    i_r: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, str]]:
    with h5.File(path, 'r') as hf:
        co_grp = hf[f'coordinates/{i_r:02d}']
        r = float(co_grp['R'][0])
        t = float(co_grp['T'][0])
        ph = np.array(co_grp['ph'][:])
        th = np.array(co_grp['th'][:])
        fields = _parse_group(hf, f'fields/{i_r:02d}')
    return r, t, ph, th, fields

def load_h5_single(path: str, dset: str) -> np.ndarray:
    with h5.File(path, 'r') as hf:
        return np.array(hf[dset])

def load_h5_many(path: str, dsets: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    with h5.File(path, 'r') as hf:
        return tuple(np.array(hf[dset]) for dset in dsets)

def apply_func_to_h5[T](
    path: str,
    dsets: tuple[tuple[str, ...], ...],
    alt: dict[str, np.ndarray],
    funcs: tuple[SurfaceFunc[T], ...],
) -> tuple[T, ...]:
    all_keys = tuple(set(sum(dsets, start=())))
    alt_args = {k: alt[k] for k in all_keys if k in alt}
    file_keys = tuple(k for k in all_keys if k not in alt)
    file_args = dict(zip(file_keys, load_h5_many(path, file_keys)))
    res = []
    for func, ds in zip(funcs, dsets):
        args = tuple(file_args[k] if k in file_keys else alt_args[k] for k in ds)
        res.append(func(*args))
    return tuple(res)
