import os
from typing import Optional, Iterable
from functools import lru_cache

import numpy as np

from simtroller.extensions.gra import scrape_dir_athdf


from .input import Input
from .athena_read import hst
from .plot2D import NativeColorPlot

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
        self.name = os.path.basename(path)
        if input_path is None:
            for file in os.listdir(path):
                if file.endswith('.inp'):
                    input_path = os.path.join(path, file)
                    break
                elif file.endswith('.par'):
                    input_path = os.path.join(path, file)
                    break
        self.input = Input(input_path)
        self.problem_id = self.input['job/problem_id']

    @property
    @lru_cache
    def hst(self) -> dict:
        return hst(f"{self.path}/{self.problem_id}.hst")


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

    def plot2d(self, time: float, *args, **kwargs) -> NativeColorPlot:
        plot = NativeColorPlot(self, *args, **kwargs)
        plot.plot(time)
        return plot

    def animate2d(self, times: Iterable[float], *args, **kwargs):
        plot = NativeColorPlot(self, *args, **kwargs)
        anim = plot.animate(times)
        return anim

def _read_ascii(path: str) -> dict:
    with open(path, 'r') as f:
        header = f.readline().split()
    keys = [hh.split(":")[1] for hh in header[1:]]

    data = np.loadtxt(path, skiprows=1, unpack=True)
    return dict(zip(keys, data))
