"""
2D plotting module for GRAthena++ simulation data.

This module provides classes and functions for creating 2D visualizations
of simulation data, including color plots, scatter plots, mesh block
overlays, and animations. It supports both native (direct from file) and
derived (computed from multiple variables) quantities.
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence, TYPE_CHECKING, Mapping
import os
from functools import lru_cache
from inspect import signature

import numpy as  np
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
if TYPE_CHECKING:
    from .simulation import Simulation

Sampling = (str | tuple[str, ...])

################################################################################


class Plot(ABC):
    """
    Abstract base class for all plot types.

    This class defines the interface for plot objects that can be updated
    at different times and optionally animated.

    Attributes:
        ax: The matplotlib Axes object for this plot.
    """

    ax: Axes

    def init_ax(self, ax: (Axes | None)):
        """
        Initialize the axes for this plot.

        Args:
            ax: Matplotlib Axes object. If None, uses the current axes.
        """
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

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

    def animate(self, *args, **kwargs):
        """
        Create an animation by calling plot() at multiple times.

        Args:
            *args: Positional arguments passed to the animate function.
            **kwargs: Keyword arguments passed to the animate function.

        Returns:
            A matplotlib FuncAnimation object.
        """
        return animate(*args, fig=self.ax.figure, plots=[self], **kwargs)

################################################################################


class Native:
    """
    Data loader for native (directly stored) simulation variables.

    This class handles loading variable data directly from simulation output
    files at specified times and samplings.

    Args:
        sim: The Simulation object to load data from.
        var: Variable name to load (can be an alias).
        sampling: Coordinate sampling specification, e.g., ('x1v', 'x2v') or 'xy'.
        strip_ghosts: If True, remove ghost zone cells from loaded data.

    Attributes:
        sim: Reference to the parent Simulation object.
        var: Full variable name (resolved from alias if needed).
        sampling: Tuple of sampling coordinate names.
        ghosts: Number of ghost zones for this variable.
        iter_range: Array of available iteration numbers.
        time_range: Array of available simulation times.
    """

    ghosts: bool

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        strip_ghosts: bool = True,
        ):
        self.sim = sim
        if isinstance(sampling, str):
            _conv = dict(x='x1v', y='x2v', z='x3v')
            sampling = tuple(_conv[s] for s in sampling)
        self.sampling = tuple(sampling)
        self.var, self.ghosts = sim.complete_var(var, self.sampling)
        self.strip_ghosts = strip_ghosts
        self.iter_range = np.arange(*sim.scrape.get_iter_range(self.var, self.sampling))
        out = sim.scrape.get_var_info(self.var, self.sampling)[0]
        self.time_range = np.array([sim.scrape.get_iter_time(out, it) for it in self.iter_range])

    @lru_cache(maxsize=1)
    def load_data(self, time: float) -> tuple:
        """
        Load simulation data at the specified time.

        Args:
            time: Simulation time to load data for. Will load the closest
                available snapshot.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of variable values
            - actual_time: The actual simulation time of the loaded snapshot
        """
        tin = time
        it = self.sim.scrape.get_iter_from_time(self.var, self.sampling, time, self.ghosts)

        out, _, dg, _ = self.sim.scrape.get_var_info(
            self.var, self.sampling, self.ghosts)

        if self.strip_ghosts:
            strip_dg = dg
        else:
            strip_dg = 0

        xyz = self.sim.scrape.get_grid(out=out, sampling=self.sampling, iterate=it, strip_dg=strip_dg)
        time = self.sim.scrape.get_iter_time(out, it)
        data = self.sim.scrape.get_var(var=self.var, sampling=self.sampling, iterate=it, strip_dg=dg)
        return xyz, data, time

    def __repr__(self):
        return f"<Native({self.var})>"

################################################################################


class Derived:
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
        strip_ghosts: If True, remove ghost zone cells from loaded data.

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
        strip_ghosts: bool = True,
        ):
        self.sim = sim
        if isinstance(sampling, str):
            _conv = dict(x='x1v', y='x2v', z='x3v')
            sampling = tuple(_conv[s] for s in sampling)
        self.sampling = tuple(sampling)
        self.var = var
        self.depends = tuple(Native(sim, dep, sampling) for dep in depends)
        self.definition = definition
        self.signature = signature(self.definition)
        self.strip_ghosts = strip_ghosts

        self.iter_range = np.array([it for it in self.depends[0].iter_range
                                    if all(np.isclose(it, dep.iter_range).any()
                                           for dep in self.depends[1:])])
        out = sim.scrape.get_var_info(self.depends[0].var, self.sampling)[0]
        self.time_range = np.array([sim.scrape.get_iter_time(out, it) for it in self.iter_range])

    @lru_cache(maxsize=1)
    def load_data(self, time: float) -> tuple:
        """
        Load and compute derived data at the specified time.

        Loads all dependent variables and applies the definition function
        to compute the derived quantity.

        Args:
            time: Simulation time to load data for.

        Returns:
            Tuple of (xyz, data, actual_time) where:
            - xyz: Tuple of coordinate arrays for each meshblock
            - data: NumPy array of computed derived values
            - actual_time: The actual simulation time of the loaded snapshot

        Raises:
            RuntimeError: If dependent variables have inconsistent time slicing.
        """

        native_data = [dep.load_data(time) for dep in self.depends]
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

################################################################################


class TimeBarPlot(Plot):
    """
    A vertical line plot that moves with time.

    Useful for indicating the current time on a separate time-series plot
    during animations.

    Args:
        ax: Matplotlib Axes object. If None, uses current axes.
        **kwargs: Keyword arguments passed to ax.axvline().

    Attributes:
        li: The matplotlib Line2D object for the vertical line.
    """

    def __init__(self, ax: (Axes | None), **kwargs):
        super().init_ax(ax=ax)
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


################################################################################


class ColorPlot(Plot, ABC):
    """
    Abstract base class for 2D color plots (pcolormesh-based).

    Provides common functionality for color plots including colorbar
    handling, axis labeling, and data transformation.

    Attributes:
        cax: Axes object for the colorbar, if any.
        cbar: Whether a colorbar is displayed.
        func: Optional function to transform data before plotting.
        kwargs: Keyword arguments for pcolormesh.
        ims: List of pcolormesh artists created.
    """

    cax: (Axes | None) = None
    cbar: bool = False

    def init_plot(
        self,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = True,
        func: Callable | None = None,
        **kwargs,
        ) -> list[Artist]:
        """
        Initialize the color plot.

        Args:
            ax: Matplotlib Axes object. If None, uses current axes.
            cbar: If True, create a colorbar. If an Axes object, use it for
                the colorbar. If False, no colorbar.
            func: Optional function to apply to data before plotting.
            **kwargs: Additional keyword arguments passed to pcolormesh.

        Returns:
            Empty list of artists (plot not yet created).
        """

        super().init_ax(ax=ax)

        if cbar is True:
            self.cbar = True
            self.cax = make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True
            self.cax = cbar

        self.ax.set_xlabel(self.sampling[0][:2])
        self.ax.set_ylabel(self.sampling[1][:2])

        self.func = func
        self.kwargs = kwargs
        self.ims = []
        return self.ims

    def make_plot(
        self,
        xyz: tuple[np.ndarray, np.ndarray],
        data: np.ndarray,
        time: float,
        var: str,
        ) -> list[Artist]:
        """
        Create the pcolormesh plot from data.

        Args:
            xyz: Tuple of (x_coords, y_coords) arrays for each meshblock.
            data: Array of data values to plot.
            time: Simulation time for the title.
            var: Variable name for the title.

        Returns:
            List of QuadMesh artists created.
        """

        if self.func is not None:
            data = self.func(data)

        self.kwargs = update_color_kwargs(var, self.kwargs, data=data)

        self.ax.set_title(f"{var} @ t= {time:.2f}")

        for fd, xx, yy in zip(data, *xyz):
            coords = np.meshgrid(xx, yy, indexing='ij')
            self.ims.append(self.ax.pcolormesh(*coords, fd, **self.kwargs))

        return self.ims

    def clean(self):
        """Remove all pcolormesh artists from the axes."""
        for im in self.ims:
            im.remove()
        self.ims = []

################################################################################


class ScatterPlot(Plot, ABC):
    """
    Abstract base class for scatter plots.

    Provides functionality for creating and updating scatter plots with
    optional color mapping.

    Attributes:
        cax: Axes object for the colorbar, if any.
        cbar: Whether a colorbar is displayed.
        scat: The PathCollection object from ax.scatter().
        kwargs: Keyword arguments for scatter.
    """

    cax: (Axes | None) = None
    cbar: bool = False
    scat = None | PathCollection

    def init_plot(
        self,
        n_points: int,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = False,
        with_c: bool = False,
        **kwargs,
        ) -> list[Artist]:
        """
        Initialize the scatter plot.

        Args:
            n_points: Number of points in the scatter plot.
            ax: Matplotlib Axes object. If None, uses current axes.
            cbar: If True, create a colorbar. If an Axes object, use it.
            with_c: If True, initialize with color values for each point.
            **kwargs: Additional keyword arguments passed to ax.scatter().

        Returns:
            List containing the scatter PathCollection artist.
        """

        super().init_ax(ax=ax)

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
        # todo: set axis labels
        self.ax.set_title(f"t = {time:.0f}")

        self.scat.set_offsets(np.column_stack(xyz))
        if c is not None:
            self.scat.set_array(c)
        return [self.scat]

################################################################################


class MeshBlockPlot(Plot, ABC):
    """
    Mixin class for adding mesh block boundary overlays to plots.

    Draws rectangles showing the boundaries of individual mesh blocks
    in adaptive mesh refinement simulations.

    Attributes:
        mb_kwargs: Keyword arguments for the Rectangle patches.
        collection: PatchCollection containing all mesh block rectangles.
    """

    def init_meshblocks(
        self,
        ax: (Axes | None) = None,
        **kwargs,
        ):
        """
        Initialize mesh block overlay rendering.

        Args:
            ax: Matplotlib Axes object. If None, uses current axes.
            **kwargs: Keyword arguments for Rectangle patches. Defaults:
                edgecolor='k', facecolor='none', linewidth=0.5, alpha=0.5.
        """

        super().init_ax(ax=ax)

        self.mb_kwargs = kwargs
        self.mb_kwargs.setdefault('edgecolor', 'k')
        self.mb_kwargs.setdefault('facecolor', 'none')
        self.mb_kwargs.setdefault('linewidth', 0.5)
        self.mb_kwargs.setdefault('alpha', 0.5)
        self.collection = PatchCollection([], match_original=True)
        self.ax.add_collection(self.collection)

    def plot_meshblocks(self, xyz) -> list[Artist]:
        """
        Draw mesh block boundaries.

        Args:
            xyz: Tuple of coordinate arrays for each meshblock.

        Returns:
            List containing the PatchCollection artist.
        """
        coll = [Rectangle((x1[0], x2[0]), x1[-1]-x1[0], x2[-1]-x2[0], **self.mb_kwargs)
                for (x1, x2) in zip(*xyz)]
        self.collection = PatchCollection(coll, match_original=True)
        self.ax.add_collection(self.collection)
        return [self.collection]

    def clean(self):
        """Remove the mesh block overlay from the axes."""
        self.collection.remove()

################################################################################


class NativeColorPlot(Native, ColorPlot, MeshBlockPlot):
    """
    2D color plot for native (directly stored) simulation variables.

    Combines data loading from Native with color plotting from ColorPlot
    and optional mesh block overlays from MeshBlockPlot.

    Args:
        sim: The Simulation object to load data from.
        var: Variable name to plot.
        sampling: Coordinate sampling, e.g., ('x1v', 'x2v') or 'xy'.
        draw_meshblocks: If True, overlay mesh block boundaries.
        meshblock_kwargs: Keyword arguments for mesh block rectangles.
        **kwargs: Additional arguments passed to ColorPlot.init_plot().

    Example:
        >>> plot = NativeColorPlot(sim, var="rho", sampling="xy")
        >>> plot.plot(time=100.0)
    """

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        draw_meshblocks: bool = False,
        meshblock_kwargs: dict = {},
        **kwargs
        ):
        super().__init__(sim, var, sampling)

        self.draw_meshblocks = draw_meshblocks
        if self.draw_meshblocks:
            self.init_meshblocks(**meshblock_kwargs)

        self.init_plot(**kwargs)
        self.ax.set_aspect('equal')

    def plot(self, time: float) -> list[Artist]:
        """
        Create or update the color plot at the given time.

        Args:
            time: Simulation time to plot.

        Returns:
            List of matplotlib artists (QuadMesh and optionally PatchCollection).
        """
        self.clean()
        xyz, data, time = self.load_data(time)

        artists = self.make_plot(xyz, data, time, self.var)

        if self.cbar:
            plt.colorbar(self.ims[-1], cax=self.cax, )
        if self.draw_meshblocks:
            artists += self.plot_meshblocks(xyz)
        return artists

    def clean(self):
        """Remove all plot elements for redrawing."""
        super().clean()
        if self.cbar:
            self.cax.clear()

################################################################################


class DerivedColorPlot(Derived, NativeColorPlot):
    """
    2D color plot for derived (computed) simulation variables.

    Like NativeColorPlot but for quantities computed from multiple
    native variables using a user-defined function.

    Args:
        sim: The Simulation object to load data from.
        var: Name for the derived variable.
        depends: Tuple of variable names that this quantity depends on.
        definition: Callable that computes the derived quantity.
        sampling: Coordinate sampling, e.g., ('x1v', 'x2v').
        draw_meshblocks: If True, overlay mesh block boundaries.
        meshblock_kwargs: Keyword arguments for mesh block rectangles.
        **kwargs: Additional arguments passed to ColorPlot.init_plot().

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
        draw_meshblocks: bool = False,
        meshblock_kwargs: dict = {},
        **kwargs
        ):
        super().__init__(sim, var, depends, definition, sampling)

        self.draw_meshblocks = draw_meshblocks
        if self.draw_meshblocks:
            self.init_meshblocks(**meshblock_kwargs)

        self.init_plot(**kwargs)
        self.ax.set_aspect('equal')

################################################################################


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
        **kwargs
    ):
        self.tracers = tracers
        n_tracers = len(tracers)
        self.init_plot(n_tracers, with_c=color_key is not None, **kwargs)
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
                if self.color_key is not None:
                    self.c[ii] = np.nan
            else:
                self.x[ii] = np.interp(time, tr_t, tr_x)
                self.y[ii] = np.interp(time, tr_t, tr_y)
                if self.color_key is not None:
                    tr_c = tr[self.color_key]
                    self.c[ii] = np.interp(time, tr_t, tr_c)

            if self.trail_len > 0:
                if (t_tr := time - self.trail_len) >= tr_t.min():
                    x_tr = np.interp(time-self.trail_len, tr_t, tr_x)
                    y_tr = np.interp(time-self.trail_len, tr_t, tr_y)
                    self.lines[ii].set_data([x_tr, self.x[ii]], [y_tr, self.y[ii]])

        return self.make_plot((self.x, self.y), c=self.c, time=time)


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
        if pbar:
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
