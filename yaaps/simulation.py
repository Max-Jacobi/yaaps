"""
Simulation module for managing GRAthena++ simulation data.

This module provides the Simulation class, which serves as the main entry point
for loading and interacting with GRAthena++ simulation output data, including
parameter files, history files, waveform data, tracer particles, and 2D plots.
"""

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
    """
    Main class for accessing GRAthena++ simulation data and creating plots.

    This class provides a unified interface for loading simulation parameters,
    accessing history data, waveform extractions, tracer particles, horizon
    data, and creating 2D visualizations.

    Args:
        path: Path to the simulation output directory.
        input_path: Optional explicit path to the input/parameter file.
            If not provided, searches for .inp or .par files in the directory.

    Attributes:
        path: Absolute path to the simulation directory.
        name: Short name derived from the directory path.
        input: Parsed Input object containing simulation parameters.
        problem_id: The problem ID from the input file.
        dx: List of grid spacings [dx1, dx2, dx3] at the finest refinement level.

    Raises:
        RuntimeError: If no valid parameter file is found in the directory.

    Example:
        >>> sim = Simulation("/path/to/simulation/output")
        >>> sim.hst["time"]  # Access history data
        array([0.0, 0.1, 0.2, ...])
        >>> sim.plot2d(time=100.0, var="rho")  # Create 2D density plot
    """
    path: str
    problem_id: str
    input: Input
    name: str
    md: dict

    def __init__(
        self,
        path: str,
        input_path: Optional[str] = None
    ):
        self.path = os.path.abspath(path)
        path_split = path.split("/")
        if (("output-" in path_split[-1]) or ("combine" in path_split[-1])) and len(path_split) > 1:
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
                raise FileNotFoundError("Could not find valid parfile")
        else:
            self.input = Input(input_path)

        self.md = {}

        rl  = 0
        if 'trackers_extrema/ref_level' in self.input:
            rl = self.input['trackers_extrema/ref_level'][0]
        else:
            for i_track in range(5):
                if f'trackers_extrema/ref_level_{i_track}' not in self.input:
                    break
                if f'trackers_extrema/ref_type_{i_track}' == 2:
                    continue
                rl = max(rl, self.input[f'trackers_extrema/ref_level_{i_track}'][0])
        self.dx: list[float] = []
        for ii in range(1, 4):
            bb = self.input[f'mesh/x{ii}max'] - self.input[f'mesh/x{ii}min']
            nx = self.input[f'mesh/nx{ii}']
            self.dx.append(bb/nx/2**rl)
        self.problem_id = self.input['job/problem_id']

    @property
    @lru_cache
    def hst(self) -> dict:
        """
        Load and return the history (.hst) file data.

        Returns:
            Dictionary mapping column names to NumPy arrays of values,
            sorted by iteration or time to remove duplicate entries.
        """
        return _straighten(hst(f"{self.path}/{self.problem_id}.hst"))

    def wav(self, radius: float, prefix="wav") -> dict:
        """
        Load waveform extraction data at a given radius.

        Args:
            radius: The extraction radius in code units.
            prefix: File prefix, defaults to "wav".

        Returns:
            Dictionary mapping column names to NumPy arrays.
        """
        path = f"{self.path}/{prefix}_r{radius:.2f}.txt"
        return _read_ascii(path)

    def tra(self, index: int, prefix="tra") -> dict:
        """
        Load tracer particle data for a given tracer index.

        Args:
            index: The tracer particle index.
            prefix: File prefix, defaults to "tra".

        Returns:
            Dictionary mapping column names to NumPy arrays.
        """
        path = f"{self.path}/{prefix}.ext{index}.txt"
        return _read_ascii(path)

    def horizon(self, index: int) -> dict:
        """
        Load apparent horizon data for a given horizon index.

        Args:
            index: The horizon index (typically 0 or 1 for binary systems).

        Returns:
            Dictionary mapping column names to NumPy arrays.
        """
        path = f"{self.path}/{self.problem_id}.horizon_summary_{index}.txt"
        return _read_ascii(path)

    @property
    @lru_cache
    def scrape(self) -> scrape_dir_athdf:
        """
        Get the directory scraper for accessing athdf output files.

        Returns:
            A scrape_dir_athdf object configured for this simulation's
            meshblock structure.
        """
        n_mb_x1 = self.input['meshblock/nx1']
        n_mb_x2 = self.input['meshblock/nx2']
        n_mb_x3 = self.input['meshblock/nx2']
        return scrape_dir_athdf(self.path, N_B=(n_mb_x1, n_mb_x2, n_mb_x3))

    @lru_cache
    def available(self, var:str) -> list:
        """
        Get available sampling and ghost zone configurations for a variable.

        Args:
            var: The variable name to query.

        Returns:
            List of (sampling, ghosts) tuples available for the variable.
        """
        return [(sampling, ghosts) for (vv, sampling, ghosts)
                in self.scrape.debug_data_keys().keys()
                if vv == var]

    def complete_var(self, var: str, sampling: tuple[str, ...]) -> tuple[str, bool]:
        """
        Complete a variable name and determine ghost zone settings.

        Takes a potentially abbreviated variable name and returns the
        full variable name along with the appropriate ghost zone setting.

        Args:
            var: Variable name or alias (e.g., "rho" for "hydro.prim.rho").
            sampling: Tuple of sampling directions, e.g., ('x1v', 'x2v').

        Returns:
            Tuple of (full_variable_name, ghost_zones).

        Raises:
            ValueError: If the variable cannot be completed uniquely or
                the requested sampling is not available.
        """
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
        """
        Create a 2D color plot of simulation data at a given time.

        Args:
            time: The simulation time to plot.
            *args: Positional arguments passed to NativeColorPlot.
            **kwargs: Keyword arguments passed to NativeColorPlot, including:
                - var: Variable name to plot (required).
                - sampling: Coordinate sampling, e.g., ('x1v', 'x2v').
                - cmap: Matplotlib colormap name.
                - norm: Normalization type ('log', 'lin', 'asinh').
                - vmin, vmax: Color scale limits.

        Returns:
            The NativeColorPlot object after plotting.
        """
        plot = NativeColorPlot(self, *args, **kwargs)
        plot.plot(time)
        return plot

    def animate2d(self, times: Iterable[float], *args, **kwargs):
        """
        Create an animation of 2D simulation data over time.

        Args:
            times: Iterable of simulation times to include in the animation.
            *args: Positional arguments passed to NativeColorPlot.
            **kwargs: Keyword arguments passed to NativeColorPlot.

        Returns:
            A matplotlib FuncAnimation object.
        """
        plot = NativeColorPlot(self, *args, **kwargs)
        anim = plot.animate(times)
        return anim


def _straighten(data: dict) -> dict:
    """
    Sort and deduplicate data dictionary by iteration or time.

    Args:
        data: Dictionary with 'iter' or 'time' key containing sortable values.

    Returns:
        Dictionary with arrays sorted and deduplicated by the sort key.
    """
    data = {k: np.atleast_1d(v) for k, v in data.items()}
    if 'iter' in data:
        dsort = data['iter']
    elif 'time' in data:
        dsort = data['time']
    else:
        return data
    if len(dsort) == 0:
        return data
    _, isort = np.unique(dsort[::-1], return_index=True)
    isort = len(dsort) - 1 - isort
    return {k: dd[isort] for k, dd in data.items()}

def _read_ascii(path: str) -> dict:
    """
    Read an ASCII data file with a header line.

    Parses files with a header format like:
    # [0]:time [1]:value1 [2]:value2 ...

    Args:
        path: Path to the ASCII file to read.

    Returns:
        Dictionary mapping column names to NumPy arrays of values,
        sorted and deduplicated by time/iteration.
    """
    with open(path, 'r') as f:
        line = "#"
        while line.startswith("#"):
            header = line
            line = f.readline()
    keys = [hh.split(":")[1] for hh in header.split()[1:]]

    # check if file is empty except header
    if os.path.getsize(path) == len(header):
        data = np.array([[]]*len(keys))
    else:
        data = np.loadtxt(path, skiprows=1, unpack=True)
    return _straighten(dict(zip(keys, data)))
