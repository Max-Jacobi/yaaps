from typing import Callable
import numpy as np

from matplotlib.colors import LogNorm, Normalize, AsinhNorm



def _update_defaults(**default) -> Callable[[dict], dict]:
    def _inner(kwargs: dict) -> dict:
        return {**default, **kwargs}
    return _inner

_color_kwargs_default: dict[str, Callable[[dict], dict]] = {
    "hydro.prim.rho": _update_defaults(cmap='magma', norm='log'),
    "passive_scalar.r_0": _update_defaults(cmap='coolwarm_r', norm='lin'), # ye
    "hydro.aux.u_t": _update_defaults(cmap='RdBu', norm='lin', vmin=-1.1, vmax=-.9), # ye
    "hydro.aux.hu_t": _update_defaults(cmap='managua', norm='lin', vmin=-1.1, vmax=-.9), # ye
    "default": _update_defaults(cmap='viridis', norm='lin'),
}

var_alias: dict[str, str] = {
    "rho": "hydro.prim.rho",
    "p": "hydro.prim.p",
    "ye": "passive_scalar.r_0",
    "s": "hydro.aux.s",
}

def update_color_kwargs(var: str, kwargs: dict, data: np.ndarray) -> dict:
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
    return kwargs
