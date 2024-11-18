from abc import ABC, abstractmethod
from typing import Callable, Iterable, TYPE_CHECKING, Mapping
from functools import lru_cache

from matplotlib import animation
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .decorations import update_color_kwargs
if TYPE_CHECKING:
    from .Simulation import Simulation

Sampling = (str | tuple[str])


class Native(ABC):
    ghosts: bool

    def init_var(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        strip_ghosts: bool = True,
        ):
        self.sim = sim
        self.sampling = tuple(sampling)
        self.var = var
        candidates = self.sim.available(self.var)
        if len(candidates) == 0:
            raise ValueError(f"Variable {var} not found in {self.sim.path}")
        for samp, ghosts in candidates:
            samp = tuple(ss+'v' for ss in samp)
            if samp == self.sampling:
                break
        else:
            raise ValueError(f"Sampling {self.sampling} not available for variable {self.var}.\n"
                             f"Available: {candidates}")
        self.ghosts = ghosts
        self.strip_ghosts = strip_ghosts

    @lru_cache(maxsize=1)
    def load_data(self, time: float) -> tuple:
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

class Plot(ABC):

    def init_ax(self, ax: (Axes | None)):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.ax.set_aspect('equal')

    @abstractmethod
    def plot(self, time: float):
        """
        Plot the data at the given time
        """
        ...

    def animate(self, times: Iterable[float], **kwargs):
        return FuncAnimation(self.ax.figure, self.plot, frames=times, **kwargs)


class ColorPlot(Plot, ABC):
    ax: Axes
    cax: (Axes | None) = None
    cbar: bool = False

    def init_plot(
        self,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = True,
        **kwargs,
        ):

        super().init_ax(ax=ax)

        if cbar is True:
            self.cbar = True
            self.cax = _make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True
            self.cax = cbar

        self.kwargs = kwargs
        self.ims = []

    def make_plot(
        self,
        xyz: tuple[np.ndarray, np.ndarray],
        data: np.ndarray,
        time: float,
        var: str,
        ):

        self.kwargs = update_color_kwargs(var, self.kwargs, vmin=data.min(), vmax=data.max())

        # todo: set axis labels

        # todo: set title
        self.ax.set_title(f"{var} @ t= {time:.2f}")


        for fd, xx, yy in zip(data, *xyz):
            coords = np.meshgrid(xx, yy, indexing='ij')
            self.ims.append(self.ax.pcolormesh(*coords, fd, **self.kwargs))


    def clean(
        self,
        ):
        for im in self.ims:
            im.remove()
        self.ims = []

class ScatterPlot(Plot, ABC):
    ax: Axes
    cax: (Axes | None) = None
    cbar: bool = False
    scat = None | PathCollection

    def init_plot(
        self,
        n_points: int,
        ax: (Axes | None) = None,
        cbar: (Axes | bool) = False,
        **kwargs,
        ):

        super().init_ax(ax=ax)

        if cbar is True:
            self.cbar = True
            self.cax = _make_cax(self.ax)
        elif isinstance(cbar, Axes):
            self.cbar = True
            self.cax = cbar

        self.kwargs = kwargs
        x = [np.nan]*n_points
        y = [np.nan]*n_points
        self.scat = self.ax.scatter(x, y, **self.kwargs)

    def make_plot(
        self,
        xyz: tuple[np.ndarray, np.ndarray],
        c: np.ndarray | None,
        time: float,
    ):
        # todo: set axis labels
        self.ax.set_title(f"t = {time:.0f}")

        if c is None:
            self.scat.set_offsets(np.column_stack(xyz))
        else:
            self.scat.set_offsets(np.column_stack(xyz))
            self.scat.set_array(c)


class MeshBlockPlot(Plot, ABC):

    def init_meshblocks(
        self,
        ax: (Axes | None) = None,
        **kwargs,
        ):

        super().init_ax(ax=ax)

        self.mb_kwargs = kwargs
        self.mb_kwargs.setdefault('edgecolor', 'k')
        self.mb_kwargs.setdefault('facecolor', 'none')
        self.mb_kwargs.setdefault('linewidth', 0.5)
        self.mb_kwargs.setdefault('alpha', 0.5)

    def plot_meshblocks(self, xyz):
        coll = [Rectangle((x1[0], x2[0]), x1[-1]-x1[0], x2[-1]-x2[0], **self.mb_kwargs)
                for (x1, x2) in zip(*xyz)]
        self.collection = PatchCollection(coll, match_original=True)
        self.ax.add_collection(self.collection)


class NativeColorPlot(Native, ColorPlot, MeshBlockPlot):

    def __init__(
        self,
        sim: "Simulation",
        var: str,
        sampling: Sampling = ('x1v', 'x2v'),
        func: Callable | None = None,
        draw_meshblocks: bool = False,
        meshblock_kwargs: dict = {},
        **kwargs
        ):
        self.init_var(sim, var, sampling)
        self.func = func

        self.draw_meshblocks = draw_meshblocks
        if self.draw_meshblocks:
            self.init_meshblocks(**meshblock_kwargs)

        self.init_plot(**kwargs)

    def plot(self, time: float):
        self.clean()
        xyz, data, time = self.load_data(time)

        if self.func is not None:
            data = self.func(data)

        self.make_plot(xyz, data, time, self.var)

        if self.cbar:
            plt.colorbar(self.ims[-1], cax=self.cax, )
        if self.draw_meshblocks:
            self.plot_meshblocks(xyz)

    def clean(self):
        super().clean()
        if self.draw_meshblocks:
            self.collection.remove()
        if self.cbar:
            self.cax.clear()

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
        self.init_plot(n_tracers, **kwargs)
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


    def plot(self, time: float):
        for ii, tr in enumerate(self.tracers):
            tr_t = tr['time'][::-1]
            tr_x = tr['x1'][::-1]
            tr_y = tr['x2'][::-1]

            if tr_t.max() <= time or time <= tr_t.min():
                self.x[ii] = np.nan
                self.y[ii] = np.nan
                if self.color_key is not None:
                    self.c[ii] = np.nan
            else:
                self.x[ii] = np.interp(time, tr_t, tr_x)
                self.y[ii] = np.interp(time, tr_t, tr_y)
                if self.color_key is not None:
                    tr_c = tr[self.color_key][::-1]
                    self.c[ii] = np.interp(time, tr_t, tr_c)

            if self.trail_len > 0:
                li_mask = (tr_t <= time) & (tr_t >= time-self.trail_len)
                self.lines[ii].set_data(tr_x[li_mask], tr_y[li_mask])

        self.make_plot((self.x, self.y), c=self.c, time=time)

def animate(
    fig: Figure,
    plots: tuple[Plot, ...],
    times: Iterable[float],
    **kwargs,
):
    def _ani(time: float):
        for plot in plots:
            plot.plot(time)
    return FuncAnimation(fig, _ani, frames=times, **kwargs)

def _make_cax(ax: Axes):
    cur_ax = plt.gca()
    pos = ax.get_position()
    right = pos.x1
    top = pos.y1
    bot = pos.y0
    cax = plt.axes([right + 0.01, bot, 0.02, top - bot])
    plt.sca(cur_ax)
    return cax
