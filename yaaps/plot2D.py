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
if TYPE_CHECKING:
    from .simulation import Simulation



def interpolate_octree_to_grid(
    octree_xyz: tuple[np.ndarray, np.ndarray], # array shape = (N_meshblocks, n_points_per_block)
    octree_data: np.ndarray,                   # array shape = (N_meshblocks, n_x_points_per_block, n_y_points_per_block)
    grid_xyz: tuple[np.ndarray, np.ndarray],   # array shape = (N_x,) and (N_y,)
    method: str = 'linear',
    ) -> np.ndarray:
    """
    Interpolate data defined on an octree mesh to a regular grid
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
    ax: Axes

    def __init__(self, ax: (Axes | None)):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

    @abstractmethod
    def plot(self, time: float) -> list[Artist]:
        """
        Plot the data at the given time
        """
        ...

    def animate(self, *args, **kwargs):
        return animate(*args, fig=self.ax.figure, plots=(self,), **kwargs)



class TimeBarPlot(Plot):

    def __init__(self, ax: (Axes | None), **kwargs):
        super().__init__(ax=ax)
        self.li = self.ax.axvline(0, **kwargs)

    def plot(self, time: float) -> list[Artist]:
        """
        animate a moving bar
        """
        self.li.set_xdata([time])
        return [self.li]



class ColorPlot[DataType: MeshData](Plot, ABC):
    cax: (Axes | None) = None
    cbar: bool = False
    data: DataType

    def __init__(
        self,
        data: DataType,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = True,
        func: Callable | None = None,
        **kwargs):

        super().__init__(ax=ax)

        self.data = data

        if cbar is True:
            self.cbar = True
            self.cax = make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True
            self.cax = cbar

        self.ax.set_xlabel(self.data.sampling[0][:2])
        self.ax.set_ylabel(self.data.sampling[1][:2])
        self.ax.set_aspect('equal')

        self.func = func
        self.kwargs = kwargs
        self.ims = []
        return self.ims

    def plot(self, time: float) -> list[Artist]:
        self.clean()
        xyz, data, time = self.data.load_data(time)

        if self.func is not None:
            data = self.func(data)

        self.kwargs = update_color_kwargs(self.data.var, self.kwargs, data=data)

        self.ax.set_title(f"{self.data.var} @ t= {time:.2f}")

        for fd, xx, yy in zip(data, *xyz):
            coords = np.meshgrid(xx, yy, indexing='ij')
            self.ims.append(self.ax.pcolormesh(*coords, fd, **self.kwargs))
        artists = self.ims.copy()

        if self.cbar:
            plt.colorbar(self.ims[-1], cax=self.cax, )
        return artists

    def clean(self):
        for im in self.ims:
            im.remove()
        self.ims = []
        if self.cax is not None:
            self.cax.clear()



class ScatterPlot(Plot, ABC):
    cax: (Axes | None) = None
    cbar: bool = False
    scat: PathCollection

    def init_plot(
        self,
        n_points: int,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = False,
        with_c: bool = False,
        **kwargs,
        ) -> list[Artist]:

        super().__init__(ax=ax)

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
        # todo: set axis labels
        self.ax.set_title(f"t = {time:.0f}")

        self.scat.set_offsets(np.column_stack(xyz))
        if c is not None:
            self.scat.set_array(c)
        return [self.scat]



class MeshBlockPlot(Plot, ABC):

    def init_meshblocks(
        self,
        ax: (Axes | None) = None,
        **kwargs,
        ):

        super().__init__(ax=ax)

        self.mb_kwargs = kwargs
        self.mb_kwargs.setdefault('edgecolor', 'k')
        self.mb_kwargs.setdefault('facecolor', 'none')
        self.mb_kwargs.setdefault('linewidth', 0.5)
        self.mb_kwargs.setdefault('alpha', 0.5)
        self.collection = PatchCollection([], match_original=True)
        self.ax.add_collection(self.collection)

    def plot_meshblocks(self, xyz) -> list[Artist]:
        coll = [Rectangle((x1[0], x2[0]), x1[-1]-x1[0], x2[-1]-x2[0], **self.mb_kwargs)
                for (x1, x2) in zip(*xyz)]
        self.collection = PatchCollection(coll, match_original=True)
        self.ax.add_collection(self.collection)
        return [self.collection]

    def clean(self):
        self.collection.remove()



class NativeColorPlot(ColorPlot[Native]):

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

        return self.make_plot((self.x, self.y), c=self.c, time=time)




class QuiverPlot(Plot):
    def __init__(
        self,
        data: VectorMeshData,
        bounds: tuple[float, float, float, float] | float,
        N_arrows: int | tuple[int, int] = 20,
        ax: (Axes | None) = None,
        grid_type: str = "cartesian",
        **kwargs,
    ):
        """Create a quiver plot for a given vector data
          Arguments:


        """
        super().__init__(ax=ax)
        self.data = data
        self.grid_type = grid_type.lower()
        self.kwargs = kwargs
        x_grid, y_grid = self._create_grid(bounds, N_arrows)
        self.grid = np.stack((x_grid.ravel(), y_grid.ravel())).T
        self.quiv = self.ax.quiver(x_grid, y_grid, np.zeros_like(x_grid), np.zeros_like(y_grid), **self.kwargs)

    def plot(self, time: float) -> list[Artist]:
        self.ax.set_title(f"{self.data.var} @ t= {time:.2f}")
        u_grid, v_grid = self.data.interp(self.grid, time=time)
        self.quiv.set_UVC(u_grid, v_grid)
        return [self.quiv]

    def _create_grid(self, bounds, N_arrows) -> tuple[np.ndarray, np.ndarray]:
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
        self.quiv.remove()



class StreamPlot(Plot):
    def __init__(
        self,
        data: VectorMeshData,
        bounds: tuple[float, float, float, float] | float,
        N_points: int | tuple[int, int] = 20,
        ax: (Axes | None) = None,
        **kwargs,
    ):
        """Create a quiver plot for a given vector data
          Arguments:


        """
        super().__init__(ax=ax)
        self.data = data
        self.kwargs = kwargs
        self.x_grid, self.y_grid = self._create_grid(bounds, N_points)
        self.grid = np.stack(np.meshgrid(self.x_grid, self.y_grid, indexing='ij'), axis=-1)
        u_grid = np.zeros((len(self.x_grid), len(self.y_grid)))
        v_grid = np.zeros((len(self.x_grid), len(self.y_grid)))
        self.stream = self.ax.streamplot(self.x_grid, self.y_grid, u_grid, v_grid, **self.kwargs)

    def plot(self, time: float) -> list[Artist]:
        self.clean()
        self.ax.set_title(f"{self.data.var} @ t= {time:.2f}")
        u_grid, v_grid = self.data.interp(self.grid, time=time)
        self.stream = self.ax.streamplot(self.x_grid, self.y_grid, u_grid, v_grid, **self.kwargs)
        return [self.stream.lines, self.stream.arrows]

    def _create_grid(self, bounds, N_arrows) -> tuple[np.ndarray, np.ndarray]:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.sca(ax)
    return cax
