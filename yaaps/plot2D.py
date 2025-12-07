"""
2D plotting module for GRAthena++ simulation data.

This module provides classes and functions for creating 2D visualizations
of simulation data, including color plots, scatter plots, quiver plots,
stream plots, mesh block overlays, and animations. It supports both native
(direct from file) and derived (computed from multiple variables) quantities.
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence, TYPE_CHECKING, Mapping
from collections.abc import Iterable
import os

import numpy as  np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm

from .decorations import update_color_kwargs
from .recipes2D import *
from .datatypes import MeshData, Native, Derived, VectorMeshData, Sampling
from .plot_formatter import PlotFormatter, PlotFormatterBase, PlotMode
if TYPE_CHECKING:
    from .simulation import Simulation


def interpolate_octree_to_grid(
    octree_xyz: tuple[np.ndarray, np.ndarray],
    octree_data: np.ndarray,
    grid_xyz: tuple[np.ndarray, np.ndarray],
    method: str = 'linear',
    ) -> np.ndarray:
    """
    Interpolate data defined on an octree mesh to a regular grid.

    Args:
        octree_xyz: Tuple of (x, y) coordinate arrays for each meshblock.
            Each array has shape (N_meshblocks, n_points_per_block).
        octree_data: Data values on the octree mesh with shape
            (N_meshblocks, n_x_points_per_block, n_y_points_per_block).
        grid_xyz: Tuple of (x_grid, y_grid) 1D arrays defining the
            regular grid to interpolate to.
        method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        2D array of interpolated data with shape (len(x_grid), len(y_grid)).
    """
    # use np.meshgrid for each meshblock to get points in 2D

    octree_x = np.empty_like(octree_data)
    octree_y = np.empty_like(octree_data)

    for i, (x, y) in enumerate(zip(*octree_xyz)):
        octree_x[i], octree_y[i] = np.meshgrid(x, y, indexing='ij')

    octree_x = octree_x.ravel()
    octree_y = octree_y.ravel()
    values = octree_data.ravel()

    points = np.column_stack((octree_x, octree_y))

    xi = np.column_stack((
        np.repeat(grid_xyz[0], len(grid_xyz[1])),
        np.tile(grid_xyz[1], len(grid_xyz[0])),
    ))
    grid_data = griddata(
        points=points,
        values=values,
        xi=xi,
        method=method,
        fill_value=0.0,
    )
    grid_data = grid_data.reshape(len(grid_xyz[0]), len(grid_xyz[1]))
    return grid_data




class Plot(ABC):
    """
    Abstract base class for all plot types.

    This class defines the interface for plot objects that can be updated
    at different times and optionally animated.

    Attributes:
        ax: The matplotlib Axes object for this plot.
        formatter: The PlotFormatterBase instance for label and unit handling.
    """

    ax: Axes
    formatter: PlotFormatterBase

    def __init__(self, ax: (Axes | None), formatter: PlotFormatterBase | str | None = None):
        """
        Initialize the plot.

        Args:
            ax: Matplotlib Axes object. If None, uses the current axes.
            formatter: PlotFormatterBase instance or str for label formatting or str
                specifying mode ("raw" or "paper"). If None, uses default
                PlotFormatter in "raw" mode.

        Raises:
            TypeError: If formatter is not a PlotFormatterBase, str, or None.
        """
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.formatter = formatter
        if isinstance(formatter, str):
            self.formatter = PlotFormatter(mode=formatter)
        elif isinstance(formatter, PlotFormatterBase):
            self.formatter = formatter
        elif formatter is None:
            self.formatter = PlotFormatter(mode="raw")
        else:
            raise TypeError("formatter must be a PlotFormatterBase, str, or None")

    def set_plot_mode(self, mode: PlotMode) -> None:
        """
        Change the plot formatting mode.

        Args:
            mode: The new formatting mode, either "raw" or "paper".

        Example:
            >>> plot.set_plot_mode("paper")  # Switch to publication mode
            >>> plot.set_plot_mode("raw")    # Switch back to raw mode
        """
        self.formatter = PlotFormatter(
            mode=mode,
            unit_converter=self.formatter.unit_converter,
            field_labels=self.formatter.field_labels,
        )

    @abstractmethod
    def plot(self, time: float) -> list[Artist]:
        """
        Plot the data at the given time.

        Args:
            time: Simulation time to plot.

        Returns:
            List of matplotlib Artist objects created or updated.
        """
        ...

    @abstractmethod
    def clean(self):
        """
        Clean up the plot by removing any created artists.
        """
        ...

    def animate(self, *args, **kwargs):
        """
        Create an animation by calling plot() at multiple times.

        Args:
            *args: Positional arguments passed to the animate function.
            **kwargs: Keyword arguments passed to the animate function.

        Returns:
            A matplotlib FuncAnimation object.
        """
        return animate(*args, fig=self.ax.figure, plots=(self,), **kwargs)



class TimeBarPlot(Plot):
    """
    A vertical line plot that moves with time.

    Useful for indicating the current time on a separate time-series plot
    during animations.

    Args:
        ax: Matplotlib Axes object. If None, uses current axes.
        formatter: PlotFormatterBase instance or str for label formatting.
        **kwargs: Keyword arguments passed to ax.axvline().

    Attributes:
        li: The matplotlib Line2D object for the vertical line.
    """

    def __init__(
        self,
        ax: (Axes | None),
        formatter: PlotFormatterBase | str | None = None,
        **kwargs
    ):
        super().__init__(ax=ax, formatter=formatter)
        self.li = self.ax.axvline(0, **kwargs)

    def plot(self, time: float) -> list[Artist]:
        """
        Update the vertical line position to the given time.

        Args:
            time: The x-position for the vertical line.

        Returns:
            List containing the updated Line2D artist.
        """
        self.li.set_xdata([time])
        return [self.li]

    def clean(self):
        """Remove the vertical line from the axes."""
        self.li.remove()


class MeshBlockPlot(Plot):
    """
    Mixin class for adding mesh block boundary overlays to plots.

    Draws rectangles showing the boundaries of individual mesh blocks
    in adaptive mesh refinement simulations.

    Attributes:
        mb_kwargs: Keyword arguments for the Rectangle patches.
        collection: PatchCollection containing all mesh block rectangles.
    """

    def __init__(
        self,
        data: MeshData,
        ax: (Axes | None) = None,
        **kwargs,
        ):
        """
        Initialize mesh block overlay rendering.

        Args:
            data: MeshData object providing meshblock coordinates.
            ax: Matplotlib Axes object. If None, uses current axes.
            formatter: PlotFormatter instance or str for label formatting. If None,
                uses a default PlotFormatter in "raw" mode.
            **kwargs: Keyword arguments for Rectangle patches. Defaults:
                edgecolor='k', facecolor='none', linewidth=0.5, alpha=0.5.
        """
        super().__init__(ax=ax)
        self.data = data

        self.mb_kwargs = kwargs
        self.mb_kwargs.setdefault('edgecolor', 'gray')
        self.mb_kwargs.setdefault('facecolor', 'none')
        self.mb_kwargs.setdefault('linewidth', 0.5)
        self.mb_kwargs.setdefault('alpha', 0.2)
        self.collection = PatchCollection([], match_original=True)
        self.ax.add_collection(self.collection)

    def plot(self, time: float) -> list[Artist]:
        """
        Draw mesh block boundaries.

        Args:
            xyz: Tuple of coordinate arrays for each meshblock.

        Returns:
            List containing the PatchCollection artist.
        """

        xyz, *_ = self.data.load_data(time) # should be lru_cached
        if self.data.sampling[0].endswith('v'):
            coll = [Rectangle(
                (1.5*x1[0] - 0.5*x1[1], 1.5*x2[0] - 0.5*x2[1]),
                x1[-1]-2*x1[0] + x1[1],
                x2[-1]-2*x2[0] + x2[1],
                **self.mb_kwargs
                ) for (x1, x2) in zip(*xyz)]
        else:
            coll = [Rectangle(
                (x1[0], x2[0]),
                x1[-1]-x1[0],
                x2[-1]-x2[0],
                **self.mb_kwargs
                ) for (x1, x2) in zip(*xyz)]
        self.collection = PatchCollection(coll, match_original=True)
        self.ax.add_collection(self.collection)
        return [self.collection]

    def clean(self):
        """Remove the mesh block overlay from the axes."""
        self.collection.remove()


class ColorPlot[DataType: MeshData](Plot, ABC):
    """
    Generic 2D color plot class for mesh data (pcolormesh-based).

    This class provides functionality for creating color plots from
    MeshData objects, including colorbar handling and data transformation.

    Type Parameters:
        DataType: The type of MeshData (Native, Derived, etc.).

    Attributes:
        cax: Axes object for the colorbar, if any.
        cbar: Whether a colorbar is displayed.
        data: The MeshData object providing the data.
        formatter: PlotFormatterBase instance for label and unit handling.
        func: Optional function to transform data before plotting.
        kwargs: Keyword arguments for pcolormesh.
        ims: List of pcolormesh artists created.
    """

    cax: (Axes | None) = None
    cbar: bool = False
    data: DataType
    mb_plot: MeshBlockPlot | None

    def __init__(
        self,
        data: DataType,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = True,
        func: Callable | None = None,
        formatter: PlotFormatterBase | str | None = None,
        draw_meshblocks: bool = False,
        **kwargs):
        """
        Initialize the color plot.

        Args:
            data: MeshData object providing the data to plot.
            ax: Matplotlib Axes object. If None, uses current axes.
            cbar: If True, create a colorbar. If an Axes object, use it for
                the colorbar. If False, no colorbar.
            func: Optional function to apply to data before plotting.
            formatter: PlotFormatterBase instance or str for label formatting. If None,
                uses a default PlotFormatter in "raw" mode.
            draw_meshblocks: If True, overlay mesh block boundaries.
            **kwargs: Additional keyword arguments passed to pcolormesh.
        """
        super().__init__(ax=ax, formatter=formatter)

        self.data = data

        if cbar is True:
            self.cbar = True
            self.cax = make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True

            self.cax = cbar
        if draw_meshblocks:
            self.mb_plot = MeshBlockPlot(ax=self.ax, data=self.data)
        else:
            self.mb_plot = None


        # Set axis labels using formatter
        x_label = self.formatter.format_axis_label(self.data.sampling[0])
        y_label = self.formatter.format_axis_label(self.data.sampling[1])
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_aspect('equal')

        self.func = func
        self.kwargs = kwargs
        self.ims = []
        return self.ims

    def plot(self, time: float) -> list[Artist]:
        """
        Create or update the color plot at the given time.

        Args:
            time: Simulation time to plot.

        Returns:
            List of QuadMesh artists created.
        """
        self.clean()
        xyz, data, actual_time = self.data.load_data(time)

        # Apply data transformation if specified
        if self.func is not None:
            data = self.func(data)

        # Convert data and coordinates based on formatter mode
        data = self.formatter.convert_data(self.data.var, data)
        converted_xyz = (
            tuple(self.formatter.convert_coordinate(self.data.sampling[0], x) for x in xyz[0]),
            tuple(self.formatter.convert_coordinate(self.data.sampling[1], y) for y in xyz[1]),
        )

        self.kwargs = update_color_kwargs(self.data.var, self.kwargs, data=data)

        # Set title using formatter
        self.ax.set_title(self.formatter.format_title(self.data.var, actual_time))

        for fd, xx, yy in zip(data, *converted_xyz):
            coords = np.meshgrid(xx, yy, indexing='ij')
            self.ims.append(self.ax.pcolormesh(*coords, fd, **self.kwargs))
        artists = self.ims.copy()

        if self.cbar:
            cbar_label = self.formatter.format_colorbar_label(self.data.var)
            cb = plt.colorbar(self.ims[-1], cax=self.cax)
            if self.formatter.mode == "paper":
                cb.set_label(cbar_label)

        if self.mb_plot is not None:
            artists += self.mb_plot.plot(time)

        return artists

    def clean(self):
        """Remove all pcolormesh artists from the axes."""
        for im in self.ims:
            im.remove()
        self.ims = []
        if self.cax is not None:
            self.cax.clear()
        if self.mb_plot is not None:
            self.mb_plot.clean()



class ScatterPlot(Plot, ABC):
    """
    Abstract base class for scatter plots.

    Provides functionality for creating and updating scatter plots with
    optional color mapping.

    Attributes:
        cax: Axes object for the colorbar, if any.
        cbar: Whether a colorbar is displayed.
        formatter: PlotFormatterBase instance for label and unit handling.
        scat: The PathCollection object from ax.scatter().
        kwargs: Keyword arguments for scatter.
    """

    cax: (Axes | None) = None
    cbar: bool = False
    scat: PathCollection

    def init_plot(
        self,
        n_points: int,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = False,
        with_c: bool = False,
        formatter: PlotFormatterBase | str | None = None,
        **kwargs,
        ) -> list[Artist]:
        """
        Initialize the scatter plot.

        Args:
            n_points: Number of points in the scatter plot.
            ax: Matplotlib Axes object. If None, uses current axes.
            cbar: If True, create a colorbar. If an Axes object, use it.
            with_c: If True, initialize with color values for each point.
            formatter: PlotFormatterBase instance or str for label formatting. If None,
                uses a default PlotFormatter in "raw" mode.
            **kwargs: Additional keyword arguments passed to ax.scatter().

        Returns:
            List containing the scatter PathCollection artist.
        """
        super().__init__(ax=ax, formatter=formatter)

        if cbar is True:
            self.cbar = True
            self.cax = make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True
            self.cax = cbar

        self.kwargs = kwargs
        x = [np.nan]*n_points
        y = [np.nan]*n_points
        if with_c:
            c = [np.nan]*n_points
            self.scat = self.ax.scatter(x, y, c=c, **self.kwargs)
        else:
            self.scat = self.ax.scatter(x, y,  **self.kwargs)
        return [self.scat]

    def make_plot(
        self,
        xyz: tuple[np.ndarray, np.ndarray],
        c: np.ndarray | None,
        time: float,
        ) -> list[Artist]:
        """
        Update the scatter plot with new positions and colors.

        Args:
            xyz: Tuple of (x, y) coordinate arrays.
            c: Optional array of color values for each point.
            time: Simulation time for the title.

        Returns:
            List containing the scatter PathCollection artist.
        """
        # Use formatter for title in paper mode, raw format otherwise
        if self.formatter.mode == "paper":
            time_scale, time_unit = self.formatter.unit_converter.get_conversion("time")
            converted_time = time * time_scale
            time_unit_clean = time_unit.strip() if time_unit else ""
            self.ax.set_title(f"$t$ = {converted_time:.2f} {time_unit_clean}")
        else:
            self.ax.set_title(f"t = {time:.0f}")

        self.scat.set_offsets(np.column_stack(xyz))
        if c is not None:
            self.scat.set_array(c)
        return [self.scat]

    def clean(self):
        """Remove the scatter plot from the axes."""
        self.scat.remove()


class NativeColorPlot(ColorPlot[Native]):
    """
    2D color plot for native (directly stored) simulation variables.

    Convenience class that creates a Native data loader and passes it
    to ColorPlot.

    Args:
        sim: The Simulation object to load data from.
        var: Variable name to plot.
        sampling: Coordinate sampling, e.g., ('x1v', 'x2v') or 'xy'.
        formatter: PlotFormatterBase instance or str for label formatting. If None,
            uses a default PlotFormatter in "raw" mode.
        **kwargs: Additional arguments passed to ColorPlot.

    Example:
        >>> plot = NativeColorPlot(sim, var="rho", sampling="xy")
        >>> plot.plot(time=100.0)
        >>> # With paper-ready formatting:
        >>> from yaaps.plot_formatter import PlotFormatter
        >>> formatter = PlotFormatter(mode="paper")
        >>> plot = NativeColorPlot(sim, var="rho", formatter=formatter)
    """

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        **kwargs
        ):
        data = Native(sim, var, sampling)
        super().__init__(data=data, **kwargs)


class DerivedColorPlot(ColorPlot[Derived]):
    """
    2D color plot for derived (computed) simulation variables.

    Convenience class that creates a Derived data loader and passes it
    to ColorPlot.

    Args:
        sim: The Simulation object to load data from.
        var: Name for the derived variable.
        depends: Tuple of variable names that this quantity depends on.
        definition: Callable that computes the derived quantity.
        sampling: Coordinate sampling, e.g., ('x1v', 'x2v').
        **kwargs: Additional arguments passed to ColorPlot.

    Example:
        >>> def velocity_magnitude(vx, vy, vz):
        ...     return np.sqrt(vx**2 + vy**2 + vz**2)
        >>> plot = DerivedColorPlot(sim, var="|v|",
        ...     depends=("velx", "vely", "velz"),
        ...     definition=velocity_magnitude)
    """

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        depends: tuple[str, ...],
        definition: Callable,
        sampling: Sampling = ('x1v', 'x2v'),
        **kwargs
        ):
        data = Derived(sim, var, depends, definition, sampling)
        super().__init__(data=data, **kwargs)



class TracerPlot(ScatterPlot):
    """
    Scatter plot for tracer particle positions over time.

    Displays tracer particles as scatter points with optional color
    mapping and trailing lines showing recent trajectory.

    Args:
        tracers: List of tracer data dictionaries, each containing at
            minimum 'time' and coordinate keys.
        coord_keys: Tuple of (x_key, y_key) specifying which tracer
            data keys to use for coordinates. Default: ('x1', 'x2').
        color_key: Optional key for color-mapping the points.
        trail_len: If > 0, draw trailing lines of this time duration.
        line_kwargs: Keyword arguments for the trailing lines.
        formatter: PlotFormatterBase instance or str for label formatting. If None,
            uses a default PlotFormatter in "raw" mode.
        **kwargs: Additional arguments passed to ScatterPlot.init_plot().

    Attributes:
        tracers: List of tracer data dictionaries.
        x, y: Current position arrays.
        c: Current color values (if color_key specified).
        lines: List of Line2D objects for trailing lines.
    """

    def __init__(
        self,
        tracers: list[Mapping],
        coord_keys: tuple[str, ...] = ('x1', 'x2'),
        color_key: str | None = None,
        trail_len: float = 0,
        line_kwargs: dict = {},
        formatter: PlotFormatterBase | str | None = None,
        **kwargs
    ):
        self.tracers = tracers
        n_tracers = len(tracers)
        self._formatter = formatter
        self.init_plot(n_tracers, with_c=color_key is not None, formatter=formatter, **kwargs)
        self.coord_keys = coord_keys
        self.color_key = color_key
        self.trail_len = trail_len
        self.line_kwargs = line_kwargs
        self.kwargs = kwargs

        self.x = np.full(n_tracers, np.nan)
        self.y = np.full(n_tracers, np.nan)
        if self.color_key is not None:
            self.c: np.ndarray | None = np.full(n_tracers, np.nan)
            if not ('norm' in self.kwargs or
                    ('vmin' in self.kwargs and 'vmax' in self.kwargs)):
                print('Warning: no normalization set for color scale')
        else:
            self.c = None

        self.lines = [self.ax.plot([], [], **self.line_kwargs)[0]
                      for _ in self.tracers]
        self.ax.set_aspect('equal')

        # Set axis labels using formatter if in paper mode
        if self.formatter.mode == "paper":
            x_coord = "x1v" if self.coord_keys[0] == "x1" else self.coord_keys[0]
            y_coord = "x2v" if self.coord_keys[1] == "x2" else self.coord_keys[1]
            self.ax.set_xlabel(self.formatter.format_axis_label(x_coord))
            self.ax.set_ylabel(self.formatter.format_axis_label(y_coord))

    def plot(self, time: float) -> list[Artist]:
        """
        Update tracer positions and trails at the given time.

        Interpolates tracer positions to the specified time and updates
        the scatter plot and trailing lines.

        Args:
            time: Simulation time to display.

        Returns:
            List of scatter and line artists.
        """
        for ii, tr in enumerate(self.tracers):
            tr_t = tr['time']
            tr_x = tr[self.coord_keys[0]]
            tr_y = tr[self.coord_keys[1]]

            if tr_t.max() < time or time < tr_t.min():
                self.x[ii] = np.nan
                self.y[ii] = np.nan
                if self.c is not None:
                    self.c[ii] = np.nan
            else:
                self.x[ii] = np.interp(time, tr_t, tr_x)
                self.y[ii] = np.interp(time, tr_t, tr_y)
                if self.c is not None:
                    tr_c = tr[self.color_key]
                    self.c[ii] = np.interp(time, tr_t, tr_c)

            if self.trail_len > 0:
                if (t_tr := time - self.trail_len) >= tr_t.min():
                    x_tr = np.interp(time-self.trail_len, tr_t, tr_x)
                    y_tr = np.interp(time-self.trail_len, tr_t, tr_y)
                    self.lines[ii].set_data([x_tr, self.x[ii]], [y_tr, self.y[ii]])

        # Convert coordinates if in paper mode
        x_coord = "x1v" if self.coord_keys[0] == "x1" else self.coord_keys[0]
        y_coord = "x2v" if self.coord_keys[1] == "x2" else self.coord_keys[1]
        converted_x = self.formatter.convert_coordinate(x_coord, self.x)
        converted_y = self.formatter.convert_coordinate(y_coord, self.y)

        return self.make_plot((converted_x, converted_y), c=self.c, time=time)

    def clean(self):
        """Remove scatter points and trailing lines from the axes."""
        self.scat.remove()
        for line in self.lines:
            line.remove()


class QuiverPlot(Plot):
    """
    Quiver (arrow) plot for vector field data.

    Displays vector data as arrows on a regular grid, supporting both
    Cartesian and polar grid layouts.

    Args:
        data: VectorMeshData object providing the vector field.
        bounds: Domain bounds. Either a single float for symmetric bounds
            (-bounds, bounds, -bounds, bounds) or a tuple (x_min, x_max, y_min, y_max)
            for Cartesian, or (r_min, r_max, phi_min, phi_max) for polar.
        N_arrows: Number of arrows per dimension. Either an int for square
            grid or tuple (N_x, N_y).
        ax: Matplotlib Axes object. If None, uses current axes.
        grid_type: Type of arrow placement grid, 'cartesian' or 'polar'.
        formatter: PlotFormatterBase instance or str for label formatting. If None,
            uses a default PlotFormatter in "raw" mode.
        **kwargs: Additional keyword arguments passed to ax.quiver().

    Attributes:
        data: The VectorMeshData providing vector values.
        grid: Array of grid points for interpolation.
        quiv: The Quiver artist.
    """

    def __init__(
        self,
        data: VectorMeshData,
        bounds: tuple[float, float, float, float] | float,
        N_arrows: int | tuple[int, int] = 20,
        ax: (Axes | None) = None,
        grid_type: str = "cartesian",
        formatter: PlotFormatterBase | str | None = None,
        **kwargs,
    ):
        super().__init__(ax=ax, formatter=formatter)
        self.data = data
        self.grid_type = grid_type.lower()
        self.kwargs = kwargs
        x_grid, y_grid = self._create_grid(bounds, N_arrows)
        self.grid = np.stack((x_grid.ravel(), y_grid.ravel())).T
        converted_x_grid = self.formatter.convert_coordinate(data.sampling[0], x_grid)
        converted_y_grid = self.formatter.convert_coordinate(data.sampling[1], y_grid)
        self.quiv = self.ax.quiver(converted_x_grid, converted_y_grid,
                                   np.zeros_like(x_grid), np.zeros_like(y_grid), **self.kwargs)

    def plot(self, time: float) -> list[Artist]:
        """
        Update the quiver plot at the given time.

        Args:
            time: Simulation time to display.

        Returns:
            List containing the Quiver artist.
        """
        # Set title using formatter
        self.ax.set_title(self.formatter.format_title(self.data.var, time))
        u_grid, v_grid = self.data.interp(self.grid, time=time)
        self.quiv.set_UVC(u_grid, v_grid)
        return [self.quiv]

    def _create_grid(self, bounds, N_arrows) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the grid of arrow positions.

        Args:
            bounds: Domain bounds specification.
            N_arrows: Number of arrows per dimension.

        Returns:
            Tuple of (x_grid, y_grid) 2D arrays.

        Raises:
            ValueError: If grid_type is not recognized.
        """
        if isinstance(N_arrows, int):
            N_x, N_y = (N_arrows, N_arrows)
        else:
            N_x, N_y = N_arrows

        if self.grid_type == "cartesian":
            if isinstance(bounds, Iterable):
                x_min, x_max, y_min, y_max = bounds
                x_grid = np.linspace(x_min, x_max, N_x)
                y_grid = np.linspace(y_min, y_max, N_y)
            else:
                x_grid = np.linspace(-bounds, bounds, N_x)
                y_grid = np.linspace(-bounds, bounds, N_y)
            x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
        elif self.grid_type == "polar":
            if isinstance(bounds, Iterable):
                r_min, r_max, ph_min, ph_max = bounds
                r_grid = np.linspace(r_min, r_max, N_x)
                ph_grid = np.linspace(ph_min, ph_max, N_y)
            else:
                r_grid = np.linspace(-bounds, bounds, N_x)
                ph_grid = np.linspace(0, 2*np.pi, N_y+1)[:-1]
            r_grid, ph_grid = np.meshgrid(r_grid, ph_grid, indexing='ij')
            cph = np.cos(ph_grid)
            x_grid = cph*r_grid
            y_grid = np.sqrt(1-cph*cph)*r_grid
        else:
            raise ValueError(f'Uknown grid_type: {self.grid_type}. Implemented are "cartesian" and "polar"')

        return x_grid, y_grid

    def clean(self):
        """Remove the quiver plot from the axes."""
        self.quiv.remove()


class StreamPlot(Plot):
    """
    Streamline plot for vector field data.

    Displays vector data as streamlines on a regular grid.

    Args:
        data: VectorMeshData object providing the vector field.
        bounds: Domain bounds. Either a single float for symmetric bounds
            (-bounds, bounds, -bounds, bounds) or a tuple (x_min, x_max, y_min, y_max).
        N_points: Number of grid points per dimension for interpolation.
            Either an int for square grid or tuple (N_x, N_y).
        ax: Matplotlib Axes object. If None, uses current axes.
        formatter: PlotFormatterBase instance or str for label formatting. If None,
            uses a default PlotFormatter in "raw" mode.
        **kwargs: Additional keyword arguments passed to ax.streamplot().

    Attributes:
        data: The VectorMeshData providing vector values.
        x_grid, y_grid: 1D arrays defining the interpolation grid.
        stream: The StreamplotSet containing lines and arrows.
    """

    def __init__(
        self,
        data: VectorMeshData,
        bounds: tuple[float, float, float, float] | float,
        N_points: int | tuple[int, int] = 20,
        ax: (Axes | None) = None,
        formatter: PlotFormatterBase | str | None = None,
        **kwargs,
    ):
        super().__init__(ax=ax, formatter=formatter)
        self.data = data
        self.kwargs = kwargs
        self.x_grid, self.y_grid = self._create_grid(bounds, N_points)
        self.converted_x_grid = self.formatter.convert_coordinate(data.sampling[0], self.x_grid)
        self.converted_y_grid = self.formatter.convert_coordinate(data.sampling[1], self.y_grid)
        self.grid = np.stack(np.meshgrid(self.x_grid, self.y_grid, indexing='ij'), axis=-1)
        u_grid = np.zeros((len(self.x_grid), len(self.y_grid)))
        v_grid = np.zeros((len(self.x_grid), len(self.y_grid)))
        self.stream = self.ax.streamplot(self.converted_x_grid, self.converted_y_grid,
                                         u_grid, v_grid, **self.kwargs)

    def plot(self, time: float) -> list[Artist]:
        """
        Update the streamline plot at the given time.

        Args:
            time: Simulation time to display.

        Returns:
            List containing the streamline and arrow artists.
        """
        self.clean()
        # Set title using formatter
        self.ax.set_title(self.formatter.format_title(self.data.var, time))
        u_grid, v_grid = self.data.interp(self.grid, time=time)
        self.stream = self.ax.streamplot(self.converted_x_grid, self.converted_y_grid,
                                         u_grid, v_grid, **self.kwargs)
        return [self.stream.lines, self.stream.arrows]

    def _create_grid(self, bounds, N_arrows) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the grid for streamline computation.

        Args:
            bounds: Domain bounds specification.
            N_arrows: Number of grid points per dimension.

        Returns:
            Tuple of (x_grid, y_grid) 1D arrays.
        """
        if isinstance(N_arrows, int):
            N_x, N_y = (N_arrows, N_arrows)
        else:
            N_x, N_y = N_arrows

        if isinstance(bounds, Iterable):
            x_min, x_max, y_min, y_max = bounds
            x_grid = np.linspace(x_min, x_max, N_x)
            y_grid = np.linspace(y_min, y_max, N_y)
        else:
            x_grid = np.linspace(-bounds, bounds, N_x)
            y_grid = np.linspace(-bounds, bounds, N_y)
        return x_grid, y_grid

    def clean(self):
        """Remove the streamline plot from the axes."""
        self.stream.lines.remove()
        for path in self.stream.arrows._paths:
            path.remove()


def animate(
    times: Sequence[float],
    fig: Figure,
    plots: tuple[Plot, ...],
    post_draw: (Callable[..., Sequence[Artist]] | None) = None,
    pbar: bool = True,
    **kwargs,
):
    """
    Create an animation from a sequence of plots over time.

    Args:
        times: Sequence of simulation times to include in the animation.
        fig: Matplotlib Figure containing the plots.
        plots: Tuple of Plot objects to update at each time.
        post_draw: Optional callable that is invoked after all plots are
            drawn. Should accept time and return a sequence of Artists.
        pbar: If True, display a progress bar during animation creation.
        **kwargs: Additional keyword arguments passed to FuncAnimation.

    Returns:
        A matplotlib FuncAnimation object.

    Example:
        >>> anim = animate(times=[0, 10, 20], fig=fig, plots=(plot1, plot2))
        >>> anim.save("animation.mp4")
    """
    if pbar:
        bar = tqdm(
            ncols=0 ,
            desc='Animating ',
            unit='frame',
            total=len(times),
            leave=False,
            )
    else:
        bar = None

    def _ani(time: float):
        artists = []
        for plot in plots:
            artists += plot.plot(time)
        if post_draw is not None:
            post_art = post_draw(time)
            try:
                artists += post_art
            except TypeError:
                ...
        if bar is not None:
            bar.update()
        return artists

    return FuncAnimation(fig, _ani, frames=times, **kwargs)


def save_frames(
    times: Sequence[float],
    fig: Figure,
    plots: tuple,
    post_draw: Callable[..., list[Artist]] | None = None,
    output_dir: str = "frames",
    prefix: str = "frame",
    dpi: int | None = None,
    pbar: bool = True,
    **savefig_kwargs,
):
    """
    Save animation frames as individual image files.

    This is an alternative to creating a video animation, useful when
    more control over individual frames is needed or when video encoding
    is not available.

    Args:
        times: Sequence of simulation times to save.
        fig: Matplotlib Figure containing the plots.
        plots: Tuple of Plot objects to update at each time.
        post_draw: Optional callable invoked after plotting each frame.
        output_dir: Directory to save frames in. Created if it doesn't exist.
        prefix: Filename prefix for frames (e.g., "frame" -> "frame0001.png").
        dpi: Resolution for saved images. If None, uses matplotlib default.
        pbar: If True, display a progress bar.
        **savefig_kwargs: Additional keyword arguments passed to fig.savefig().

    Returns:
        List of paths to the saved frame files.

    Example:
        >>> frames = save_frames(times, fig, plots, output_dir="my_frames")
        >>> print(frames[0])
        'my_frames/frame0000.png'
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(times)
    todo = enumerate(times)

    def bar(*args, **kwargs):
        kwargs = {**dict(total=total, desc="Saving frames", ncols=0,
                         unit="frame", leave=False, disable=not pbar),
                  **kwargs}
        return tqdm(*args, **kwargs)

    def work(it):
        i, t = it
        artists = []
        for plot in plots:
            artists.extend(plot.plot(t))

        if post_draw is not None:
            extra = post_draw(t)
            # allow for single-Artist return
            if isinstance(extra, list):
                artists.extend(extra)
            elif extra is not None:
                artists.append(extra)

        fname = os.path.join(output_dir, f"{prefix}{i:04d}.png")
        fig.savefig(fname, dpi=dpi, **savefig_kwargs)
        return fname

    return list(bar(map(work, todo)))


def make_cax(ax: Axes):
    """
    Create a colorbar axes adjacent to an existing axes.

    Args:
        ax: The main axes to attach the colorbar to.

    Returns:
        A new Axes object positioned to the right of the main axes,
        suitable for use as a colorbar axes.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.sca(ax)
    return cax
