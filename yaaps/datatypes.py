"""
Data types module for mesh-based simulation data.

This module provides abstract and concrete classes for loading and
interpolating simulation data from GRAthena++ output files. It supports
native (directly stored) variables, derived (computed) quantities,
and vector field data.
"""

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
from functools import lru_cache
from inspect import signature

import numpy as np
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from .simulation import Simulation

Sampling = tuple[str, str] | str
"""Type alias for coordinate sampling specification.

Can be either a tuple of coordinate names like ('x1v', 'x2v') or
a shorthand string like 'xy' which gets converted automatically.
"""


_n_ghosts = {"nearest": 0, "linear": 1, "slinear": 1, "cubic": 2, "quintic": 3, "pchip": 2}
"""Mapping from interpolation method to required ghost zone count."""


class MeshData(ABC):
    """
    Abstract base class for mesh-based data loaders.

    This class defines the interface for loading and interpolating
    simulation data on adaptive mesh refinement grids.

    Attributes:
        sim: Reference to the parent Simulation object.
        var: Variable name.
        sampling: Tuple of sampling coordinate names, e.g., ('x1v', 'x2v').
        iter_range: Array of available iteration numbers.
        time_range: Array of available simulation times.
    """

    sim: "Simulation"
    var: str
    sampling: tuple[str, str]
    iter_range: np.ndarray
    time_range: np.ndarray

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        ):
        """
        Initialize the MeshData object.

        Args:
            sim: The Simulation object to load data from.
            var: Variable name.
            sampling: Coordinate sampling specification, e.g., ('x1v', 'x2v') or 'xy'.
        """
        self.sim = sim
        if isinstance(sampling, str):
            _conv = dict(x='x1v', y='x2v', z='x3v')
            self.sampling = (_conv[sampling[0]], _conv[sampling[1]])
        else:
            self.sampling = sampling
        self.var = var

    @abstractmethod
    @lru_cache(maxsize=1)
    def load_data(self, time: float) -> tuple:
        """
        Load data at the given time.

        Args:
            time: Simulation time to load data for.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of variable values
            - actual_time: The actual simulation time of the loaded snapshot
        """
        ...

    def interp(
        self,
        points: np.ndarray,
        time: float,
        method: str = 'linear',
        ) -> np.ndarray:
        """
        Interpolate data at given points and time.

        Args:
            points: (N, 2) array of points to interpolate at.
            time: Simulation time to load data at.
            method: Interpolation method, one of 'nearest', 'linear',
                'slinear', 'cubic', 'quintic', 'pchip'.

        Returns:
            (N,) array of interpolated data values. Points outside the
            domain are set to NaN.
        """
        points = np.atleast_2d(points)
        point_shape = points.shape[:-1]
        assert points.shape[-1] == 2
        points = points.reshape(-1, 2)

        xp, yp = points.T

        (block_x, block_y), block_data, time = self.load_data(time, strip_ghosts=False)

        n_ghosts = _n_ghosts[method]

        xmin, xmax = block_x[:, [n_ghosts, -n_ghosts]].T
        ymin, ymax = block_y[:, [n_ghosts, -n_ghosts]].T

        mask = (
              (xp[None] >= xmin[:, None]) &
              (xp[None] <= xmax[:, None]) &
              (yp[None] >= ymin[:, None]) &
              (yp[None] <= ymax[:, None])
            )

        idx = mask.argmax(axis=0)
        un_idx = np.unique(idx)

        interp_data  = np.empty_like(xp)

        for ii in un_idx:
            msk = (idx == ii)
            x = block_x[ii]
            y = block_y[ii]
            data = block_data[ii]

            interp = RegularGridInterpolator((x, y), data, bounds_error=True, method=method)

            interp_data[msk] = interp(points[msk])

        interp_data[~mask.any(axis=0)] = np.nan
        return interp_data.reshape(point_shape)



class Native(MeshData):
    """
    Data loader for native (directly stored) simulation variables.

    This class handles loading variable data directly from simulation output
    files at specified times and samplings.

    Args:
        sim: The Simulation object to load data from.
        var: Variable name to load (can be an alias).
        sampling: Coordinate sampling specification, e.g., ('x1v', 'x2v') or 'xy'.

    Attributes:
        sim: Reference to the parent Simulation object.
        var: Full variable name (resolved from alias if needed).
        sampling: Tuple of sampling coordinate names.
        ghosts: Number of ghost zones for this variable.
        iter_range: Array of available iteration numbers.
        time_range: Array of available simulation times.
    """

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        ):
        super().__init__(sim, var, sampling)
        self.var, self.ghosts = sim.complete_var(var, self.sampling)

        self.iter_range = np.arange(*sim.scrape.get_iter_range(self.var, self.sampling))
        out = sim.scrape.get_var_info(self.var, self.sampling)[0]
        self.time_range = np.array([sim.scrape.get_iter_time(out, it) for it in self.iter_range])

    @lru_cache(maxsize=1)
    def load_data(self, time: float, strip_ghosts: bool = True) -> tuple:
        """
        Load simulation data at the specified time.

        Args:
            time: Simulation time to load data for. Will load the closest
                available snapshot.
            strip_ghosts: If True, remove ghost zone cells from loaded data.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of variable values
            - actual_time: The actual simulation time of the loaded snapshot
        """
        it = self.sim.scrape.get_iter_from_time(self.var, self.sampling, time, self.ghosts)

        out, _, dg, _ = self.sim.scrape.get_var_info(
            self.var, self.sampling, self.ghosts)

        if strip_ghosts:
            strip_dg = dg
        else:
            strip_dg = 0

        xyz = self.sim.scrape.get_grid(out=out, sampling=self.sampling, iterate=it, strip_dg=strip_dg)
        time = self.sim.scrape.get_iter_time(out, it)
        data = self.sim.scrape.get_var(var=self.var, sampling=self.sampling, iterate=it, strip_dg=strip_dg)
        return xyz, data, time

    def __repr__(self):
        return f"<Native({self.var})>"



class Derived(MeshData):
    """
    Data loader for derived (computed) simulation variables.

    This class handles computing derived quantities from multiple native
    variables using a user-defined function.

    Args:
        sim: The Simulation object to load data from.
        var: Name for the derived variable.
        depends: Tuple of variable names that this derived quantity depends on.
        definition: Callable that computes the derived quantity from the
            dependent variables. Can optionally accept 'xyz', 'time', and
            'sampling' keyword arguments.
        sampling: Coordinate sampling specification, e.g., ('x1v', 'x2v').

    Attributes:
        sim: Reference to the parent Simulation object.
        var: Name of the derived variable.
        depends: Tuple of Native objects for each dependency.
        definition: The computation function.
        iter_range: Array of iteration numbers where all dependencies are available.
        time_range: Array of simulation times where all dependencies are available.
    """

    ghosts: bool

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        depends: tuple[str, ...],
        definition: Callable,
        sampling: Sampling = ('x1v', 'x2v'),
        ):
        super().__init__(sim, var, sampling)

        self.depends = tuple(Native(sim, dep, sampling) for dep in depends)
        self.definition = definition
        self.signature = signature(self.definition)

        self.iter_range = np.array([it for it in self.depends[0].iter_range
                                    if all(np.isclose(it, dep.iter_range).any()
                                           for dep in self.depends[1:])])
        out = sim.scrape.get_var_info(self.depends[0].var, self.sampling)[0]
        self.time_range = np.array([sim.scrape.get_iter_time(out, it) for it in self.iter_range])

    @lru_cache(maxsize=1)
    def load_data(self, time: float, strip_ghosts: bool = True) -> tuple:
        """
        Load and compute derived data at the specified time.

        Loads all dependent variables and applies the definition function
        to compute the derived quantity.

        Args:
            time: Simulation time to load data for.
            strip_ghosts: If True, remove ghost zone cells from loaded data.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of computed derived values
            - actual_time: The actual simulation time of the loaded snapshot

        Raises:
            RuntimeError: If dependent variables have inconsistent time slicing.
        """
        native_data = [dep.load_data(time, strip_ghosts) for dep in self.depends]
        xyzs = [nd[0] for nd in native_data]
        datas = [nd[1] for nd in native_data]
        times = [nd[2] for nd in native_data]

        xyz = xyzs[0]
        time = times[0]
        if not all(np.isclose(time, t) for t in times[1:]):
            raise RuntimeError(f"Not all dependencies for {self.var} have the same time slicing!")
        kwargs = {name: value for name, value in (('xyz', xyz), ('time', time), ('sampling', self.sampling))
                  if name in self.signature.parameters}
        data = self.definition(*datas, **kwargs)
        return xyz, data, time

    def __repr__(self):
        return f"<Derived({self.var})>"



class VectorMeshData(MeshData):
    """
    Data loader for vector field data on mesh grids.

    This class combines multiple scalar MeshData components into a vector
    field, handling loading and interpolation of all components together.

    Attributes:
        sim: Reference to the parent Simulation object.
        var: Comma-separated string of component variable names.
        sampling: Tuple of sampling coordinate names.
        components: Tuple of two MeshData objects for the 2D vector components.
        iter_range: Array of iteration numbers where all components are available.
        time_range: Array of simulation times where all components are available.
    """

    components: tuple[MeshData, MeshData]

    def __init__(
        self,
        components: tuple[MeshData, MeshData, MeshData],
    ):
        """
        Initialize VectorMeshData from three MeshData components.

        Args:
            components: Tuple of three MeshData objects representing the
                x, y, and z vector components. The appropriate two components
                are selected based on the sampling specification.

        Raises:
            RuntimeError: If components don't share the same simulation or sampling.

        Note:
            Alternatively, use from_native() class method to create from
            Native objects given the variable names.
        """
        first = components[0]
        if not all(comp.sim.path == first.sim.path for comp in components):
            raise RuntimeError("Vector components do not belong to the same simulation!")
        self.sim = first.sim

        if not all(comp.sampling == first.sampling for comp in components):
            raise RuntimeError("Vector components do not share the same sampling!")
        self.sampling = first.sampling

        # pick components based on sampling
        sampling_idx = [int(s[1]) - 1 for s in self.sampling]
        self.components = (components[sampling_idx[0]], components[sampling_idx[1]])

        self.var = ",".join(comp.var for comp in components)

        self.iter_range = np.array([it for it in self.components[0].iter_range
                                    if all(np.isclose(it, dep.iter_range).any()
                                           for dep in self.components[1:])])
        out = self.sim.scrape.get_var_info(self.components[0].var, self.sampling)[0]
        self.time_range = np.array([self.sim.scrape.get_iter_time(out, it) for it in self.iter_range])

    @lru_cache(maxsize=1)
    def load_data(self, time: float) -> tuple:
        """
        Load vector data at the specified time.

        Args:
            time: Simulation time to load data for.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of shape (2, ...) containing vector components
            - actual_time: The actual simulation time of the loaded snapshot

        Raises:
            RuntimeError: If vector components don't share the same grid or time.
        """
        native_data = [comp.load_data(time) for comp in self.components]

        xyzs = [nd[0] for nd in native_data]
        datas = [nd[1] for nd in native_data]
        times = [nd[2] for nd in native_data]

        xyz0 = xyzs[0]
        time0 = times[0]

        if not np.allclose(xyz0, xyzs[1]):
            raise RuntimeError("Vector components do not share the same grid!")

        if not all(np.isclose(time0, t) for t in times[1:]):
            raise RuntimeError("Vector components do not share the same time!")

        data = np.stack(datas, axis=0)

        return xyz0, data, time0

    def interp(
        self,
        points: np.ndarray,
        time: float,
        method: str = 'linear',
        ) -> np.ndarray:
        """
        Interpolate vector data at given points and time.

        Args:
            points: (N, 2) or (..., 2) array of points to interpolate at.
            time: Simulation time to load data at.
            method: Interpolation method, one of 'nearest', 'linear',
                'slinear', 'cubic', 'quintic', 'pchip'.

        Returns:
            (2, ...) array of interpolated vector data, where the first
            dimension corresponds to the vector components.
        """
        interp_components = [comp.interp(points, time, method) for comp in self.components]
        interp_data = np.stack(interp_components, axis=0)
        return interp_data

    def __repr__(self):
        vars_ = ", ".join(comp.var for comp in self.components)
        return f"<VectorMeshData({vars_})>"

    @classmethod
    def from_native(
        cls,
        sim: "Simulation",
        vars: tuple[str, str, str],
        sampling: Sampling = ("x1v", "x2v"),
    ) -> "VectorMeshData":
        """
        Create a VectorMeshData from native variable names.

        This is a convenience method for creating vector data from
        three native variable names without manually creating Native objects.

        Args:
            sim: The Simulation object to load data from.
            vars: Tuple of three variable names for the x, y, z components.
            sampling: Coordinate sampling specification.

        Returns:
            A new VectorMeshData object.

        Example:
            >>> vec = VectorMeshData.from_native(sim, ("velx", "vely", "velz"))
        """
        vx, vy, vz = (Native(sim, v, sampling) for v in vars)
        return cls((vx, vy, vz))
