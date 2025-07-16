import os
from typing import Optional, Iterable
from functools import lru_cache

import numpy as np

from simtroller.extensions.gra import scrape_dir_athdf


from .input import Input
from .athena_read import hst
from .plot2D import NativeColorPlot
from .decorations import var_alias

class Simulation:
    path: str
    problem_id: str
    input: Input
    name: str

    def __init__(
        self,
        path: str,
        input_path: Optional[str] = None
    ):
        self.path = path
        path_split = path.split("/")
        if ("output-" in path_split[-1]) or ("combine" in path_split[-1]):
            self.name = path_split[-2]
        else:
            self.name = path_split[-1]
        if input_path is None:
            for file in os.listdir(path):
                if file.endswith('.inp') or file.endswith('.par'):
                    try:
                        self.input = Input(os.path.join(path, file))
                        break
                    except ValueError:
                        ...
            else:
                raise RuntimeError("Could not find valid parfile")
        else:
            self.input = Input(input_path)


        rl = self.input[f'trackers_extrema/ref_level'][0]
        self.dx: list[float] = []
        for ii in range(1, 4):
            bb = self.input[f'mesh/x{ii}max'] - self.input[f'mesh/x{ii}min']
            nx = self.input[f'mesh/nx{ii}']
            self.dx.append(bb/nx/2**rl)
        self.problem_id = self.input['job/problem_id']

    @property
    @lru_cache
    def hst(self) -> dict:
        return _straighten(hst(f"{self.path}/{self.problem_id}.hst"))

    def wav(self, radius: float, prefix="wav") -> dict:
        path = f"{self.path}/{prefix}_r{radius:.2f}.txt"
        return _read_ascii(path)

    def tra(self, index: int, prefix="tra") -> dict:
        path = f"{self.path}/{prefix}.ext{index}.txt"
        return _read_ascii(path)

    def horizon(self, index: int) -> dict:
        path = f"{self.path}/{self.problem_id}.horizon_summary_{index}.txt"
        return _read_ascii(path)

    @property
    @lru_cache
    def scrape(self) -> scrape_dir_athdf:
        n_mb_x1 = self.input['meshblock/nx1']
        n_mb_x2 = self.input['meshblock/nx2']
        n_mb_x3 = self.input['meshblock/nx2']
        return scrape_dir_athdf(self.path, N_B=(n_mb_x1, n_mb_x2, n_mb_x3))

    @lru_cache
    def available(self, var:str) -> list:
        return [(sampling, ghosts) for (vv, sampling, ghosts)
                in self.scrape.debug_data_keys().keys()
                if vv == var]

    def complete_var(self, var: str, sampling: tuple[str, ...]) -> tuple[str, int]:
        if var in var_alias:
            var = var_alias[var]
        sampling = tuple(x[:2] for x in sampling)
        candidates = [(vv, samp, gh)
                      for vv, samp, gh in self.scrape.debug_data_keys().keys()
                      if vv.endswith(var)]
        if len(candidates) == 0:
            raise ValueError(f"Can't complete variable {var}.")
        vcan = [(vv, samp, gh) for vv, samp, gh in candidates if samp==sampling]
        if len(vcan) > 1:
            raise ValueError(f"More than one completion of {var} available: {[v for v, *_ in vcan]}")
        if len(vcan) == 0:
            raise ValueError(f"Sampling {sampling} not available for variable {var}.\n"
                             f"Available: {[(v, s) for v, s, _ in candidates]}")
        return vcan[0][0], vcan[0][2]

    def plot2d(self, time: float, *args, **kwargs) -> NativeColorPlot:
        plot = NativeColorPlot(self, *args, **kwargs)
        plot.plot(time)
        return plot

    def animate2d(self, times: Iterable[float], *args, **kwargs):
        plot = NativeColorPlot(self, *args, **kwargs)
        anim = plot.animate(times)
        return anim

def _straighten(data: dict) -> dict:
    if 'iter' in data:
        dsort = data['iter']
    elif 'time' in data:
        dsort = data['time']
    else:
        return data
    _, isort = np.unique(dsort, return_index=True)
    return {k: np.atleast_1d(dd)[isort] for k, dd in data.items()}

def _read_ascii(path: str) -> dict:
    with open(path, 'r') as f:
        line = "#"
        while line.startswith("#"):
            header = line
            line = f.readline()
    keys = [hh.split(":")[1] for hh in header.split()[1:]]

    data = np.loadtxt(path, skiprows=1, unpack=True)
    return _straighten(dict(zip(keys, data)))
