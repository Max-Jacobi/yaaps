import os
from collections.abc import Mapping
from typing import Sequence, Iterable
from functools import cached_property, cached_property

import numpy as np
from tabulatedEOS.PyCompOSEEOS import PyCompOSEEOS

from ..parallel_utils import do_parallel, do_parallel_enumerate
from .h5_utils import apply_func_to_h5, parse_h5, load_h5_many, load_h5_single
from .surface_func import SurfaceFunc, dA_d, get_func, Vector_d, AnyField

class Surfaces(Mapping):

    def __init__(
        self,
        paths: Iterable[str],
        i_surface: int,
        i_radius: int,
        n_cpu: int = 1,
        verbose: bool = False,
        eos_path: None | str = None
        ):
        self.paths = paths
        self.i_s = int(i_surface)
        self.i_r = int(i_radius)
        self.n_cpu = n_cpu
        self.verbose = verbose
        if eos_path is None:
            self.eos = None
        else:
            self.eos = PyCompOSEEOS(eos_path, code_units="GeometricSolar")
        self._parse_files()

    def _parse_files(self):
        times = []
        self.r = np.nan
        self.files = np.array([f'{path}/{f}'
                               for path in self.paths
                               for f in os.listdir(path)
                               if f".surface{self.i_s}." in f])

        for _r, _t, _ph, _th, _fields in do_parallel(
            parse_h5,
            self.files,
            n_cpu=self.n_cpu,
            verbose=self.verbose,
            desc='Parsing files',
            unit='files',
            args=(self.i_r, ),
            ordered=True,
            ):
            if np.isnan(self.r):
                self.r = _r
                ph = _ph
                th = _th
                self.fields = _fields
            else:
                assert np.isclose(_r, self.r), 'detected changing R'
                assert np.isclose(_ph, ph).all(), 'detected changing phi'
                assert np.isclose(_th, th).all(), 'detected changing theta'
                assert all(k in self.fields for k in _fields), 'detected changing fields'
            times.append(_t)
        isort = np.argsort(times)
        self.files = self.files[isort]
        self.times = np.array(times)[isort]
        dth, dph = th[1] - th[0], ph[1] - ph[0]
        m_th, m_ph = np.meshgrid(th, ph, indexing='ij')
        dA = self.r * self.r * np.sin(m_th) * dth * dph
        x = self.r * np.sin(m_th) * np.cos(m_ph)
        y = self.r * np.sin(m_th) * np.sin(m_ph)
        z = self.r * np.cos(m_th)
        r = np.full_like(x, self.r)

        self.aux = {"x": x, "y": y, "z": z,
                    "r": r, "ph": m_ph, "th": m_th,
                    "dA": dA}

        self.shape = (len(self.times), *x.shape)

    @cached_property
    def dA_d(self) -> tuple[Vector_d, ...]:
        return tuple(dA[0] for dA in self.process_h5_parallel((dA_d, )))

    def complete_input(self, key: str):
        return self.fields.get(key, key)

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def _getsingle(self, key: str) -> AnyField:
        if key not in self.fields and key not in self.aux:
            raise KeyError(f"{key} not in {self}")

        if key in self.aux:
            return self.aux[key]
        ar = np.empty(self.shape)
        for it, _dat in do_parallel_enumerate(
            load_h5_single,
            self.files,
            n_cpu=self.n_cpu,
            args=(self.complete_input(key), ),
            verbose=self.verbose,
            desc=f'Loading {key}',
            unit='files',
            ):
                ar[it] = _dat
        return ar

    def _getmany(self, keys: Sequence[str]) -> dict[str, AnyField]:
        if any(missing := [(key not in self.fields and key not in self.aux) for key in keys]):
            raise ValueError(f"{[kk for kk, miss in zip(keys, missing) if miss]} not in {self}")
        ar = {k: np.empty(self.shape) for k in keys}
        for it, _dat in do_parallel_enumerate(
            load_h5_many,
            self.files,
            n_cpu=self.n_cpu,
            args=([self.complete_input(k) for k in keys if k not in self.aux], ),
            verbose=self.verbose,
            desc=f'Loading {keys}',
            unit='files',
            ):
                for k, d in zip(keys, _dat):
                    ar[k][it] = d
        for k in keys:
            if k in self.aux:
                ar[k] = self.aux[k]
        return ar

    def __getitem__(self, keys: (str | Sequence[str])) -> (AnyField | dict[str, AnyField]):
        if isinstance(keys, str):
            return self._getsingle(keys)
        return self._getmany(keys)

    def __str__(self) -> str:
        return f"Surface(s{self.i_s:d} r{self.i_r:d})"

    def __repr__(self) -> str:
        return str(self)

    def process_h5_parallel(
        self,
        funcs: tuple[(SurfaceFunc | str), ...],
        ordered: bool = False,
        ) -> Iterable[tuple]:
        _funcs = tuple(get_func(f) for f in funcs)
        inputs = tuple(tuple(self.complete_input(k) for k in func.keys) for func in _funcs)
        return do_parallel(
            apply_func_to_h5,
            self.files,
            n_cpu=self.n_cpu,
            args=(inputs, self.aux, _funcs,),
            verbose=self.verbose,
            desc=f'Calculating {[func.name for func in _funcs]}:',
            unit='files',
            ordered=ordered,
            )

    def sphere_flux(self, v_u: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([np.sum(v*dA) for v, dA in zip(v_u, self.dA_d)])
