"""
Decorations module for handling color and normalization settings for plots.

This module provides utilities to configure matplotlib colormap and normalization
settings for various variables commonly used in GRAthena++ simulations.
"""

from typing import Callable
import numpy as np

from matplotlib.colors import LogNorm, Normalize, AsinhNorm


def _update_defaults(**default) -> Callable[[dict], dict]:
    """
    Create a function that merges default keyword arguments with provided ones.

    Args:
        **default: Default keyword arguments to use.

    Returns:
        A function that takes a dict of kwargs and returns a merged dict
        with the defaults applied (provided kwargs take precedence).
    """
    def _inner(kwargs: dict) -> dict:
        return {**default, **kwargs}
    return _inner

_color_kwargs_default: dict[str, Callable[[dict], dict]] = {
    "hydro.prim.rho": _update_defaults(cmap='magma', norm='log'),
    "passive_scalar.r_0": _update_defaults(cmap='coolwarm_r', norm='lin'), # ye
    "hydro.aux.u_t": _update_defaults(cmap='RdBu', norm='lin', vmin=-1.1, vmax=-.9),
    "hydro.aux.hu_t": _update_defaults(cmap='managua', norm='lin', vmin=-1.1, vmax=-.9),
    "M1.lab.sc_E_00": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.lab.sc_E_01": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.lab.sc_E_02": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.lab.sc_nG_00": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "M1.lab.sc_nG_01": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "M1.lab.sc_nG_02": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "M1.rad.sc_J_00": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.rad.sc_J_01": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.rad.sc_J_02": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "M1.rad.sc_n_00": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "M1.rad.sc_n_01": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "M1.rad.sc_n_02": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "default": _update_defaults(cmap='viridis', norm='lin'),
}

var_alias: dict[str, str] = {
    "rho": "hydro.prim.rho",
    "p": "hydro.prim.p",
    "ye": "passive_scalar.r_0",
    "s": "hydro.aux.s",
    "m1_E_e": "M1.lab.sc_E_00",
    "m1_E_a": "M1.lab.sc_E_01",
    "m1_E_x": "M1.lab.sc_E_02",
    "m1_nG_e": "M1.lab.sc_nG_00",
    "m1_nG_a": "M1.lab.sc_nG_01",
    "m1_nG_x": "M1.lab.sc_nG_02",
    "m1_J_e": "M1.rad.sc_J_00",
    "m1_J_a": "M1.rad.sc_J_01",
    "m1_J_x": "M1.rad.sc_J_02",
    "m1_n_e": "M1.rad.sc_n_00",
    "m1_n_a": "M1.rad.sc_n_01",
    "m1_n_x": "M1.rad.sc_n_02",
}

def update_color_kwargs(var: str, kwargs: dict, data: np.ndarray) -> dict:
    """
    Update color-related keyword arguments based on the variable and data.

    This function applies default colormap and normalization settings based on
    the variable name, then computes appropriate vmin/vmax values if not provided.

    Args:
        var: Variable name used to look up default color settings.
        kwargs: Dictionary of keyword arguments to update.
        data: NumPy array of data values used to compute vmin/vmax if not provided.

    Returns:
        Updated dictionary of keyword arguments with colormap and normalization
        settings applied. The 'norm' key will contain a matplotlib Normalize,
        LogNorm, or AsinhNorm instance.

    Raises:
        ValueError: If an unknown normalization type is specified.
    """
    if var in _color_kwargs_default:
        kwargs = _color_kwargs_default[var](kwargs)
    else: # default
        kwargs = _color_kwargs_default['default'](kwargs)

    if 'norm' not in kwargs:
        kwargs['norm'] = 'lin'

    if isinstance(kwargs['norm'], str):
        norm = kwargs.pop('norm', 'lin')
        fdata = data.copy()
        if 'vmin' not in kwargs or 'vmax' not in kwargs:
            fdata = fdata[np.isfinite(data)]
        if norm == 'log':
            fdata = fdata[fdata>0]

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = fdata.min()

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = fdata.max()

        if norm == 'log':
            kwargs['norm'] = LogNorm(vmin=vmin, vmax=vmax)
        elif norm == 'lin':
            kwargs['norm'] = Normalize(vmin=vmin, vmax=vmax)
        elif norm == 'asinh':
            absvmax = max(abs(vmax), abs(vmin))
            lin_width = kwargs.pop('linear_width', absvmax*1e-7)
            kwargs['norm'] = AsinhNorm(linear_width=lin_width, vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f'Unknown norm: {norm}')
    else:
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
    return kwargs
