from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING
from functools import lru_cache

from matplotlib import animation
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation

from .decorations import update_color_kwargs
if TYPE_CHECKING:
    from .Simulation import Simulation

Sampling = (str | Iterable[str])


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

    def init_ax(self, ax: (plt.Axes | None)):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        # self.ax.set_aspect('equal')

    @abstractmethod
    def plot(self, time: float):
        """
        Plot the data at the given time
        """
        ...


class ColorPlot(Plot, ABC):
    ax: plt.Axes
    cax: (plt.Axes | None) = None
    cbar: bool = False

    def init_plot(
        self,
        ax: (plt.Axes | None) = None,
        cbar: (plt.Axes | bool) = True,
        **kwargs,
        ):

        super().init_ax(ax=ax)

        if cbar is True:
            self.cbar = True
            self.cax = _make_cax(self.ax)
        elif isinstance(cbar, plt.Axes):
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

        for fd, xx, yy in zip(data, *xyz):
            coords = np.meshgrid(xx, yy, indexing='ij')
            self.ims.append(self.ax.pcolormesh(*coords, fd, **self.kwargs))


    def clean(
        self,
        ):
        for im in self.ims:
            im.remove()
        self.ims = []

class MeshBlockPlot(Plot, ABC):

    def init_meshblocks(
        self,
        ax: (plt.Axes | None) = None,
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
        draw_meshblocks: bool = False,
        meshblock_kwargs: dict = {},
        **kwargs
        ):
        self.init_var(sim, var, sampling)

        self.draw_meshblocks = draw_meshblocks
        if self.draw_meshblocks:
            self.init_meshblocks(**meshblock_kwargs)

        self.init_plot(**kwargs)


    def plot(self, time: float):
        xyz, data, time = self.load_data(time)

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

    def animate(self, times: Iterable[float], **kwargs):
        def _anim(time):
            self.clean()
            self.plot(time)
            return self.ims
        animation = FuncAnimation(self.ax.figure, _anim, frames=times, **kwargs)
        return animation

def _make_cax(ax: plt.Axes, **kwargs):
    cur_ax = plt.gca()
    pos = ax.get_position()
    right = pos.x1
    top = pos.y1
    bot = pos.y0
    cax = plt.axes([right + 0.01, bot, 0.02, top - bot])
    plt.sca(cur_ax)
    return cax
