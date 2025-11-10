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
    "hydro.aux.u_t": _update_defaults(cmap='RdBu', norm='lin', vmin=-1.1, vmax=-.9),
    "hydro.aux.hu_t": _update_defaults(cmap='managua', norm='lin', vmin=-1.1, vmax=-.9),
    "m1.lab.sc_E_00": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.lab.sc_E_01": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.lab.sc_E_02": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.lab.sc_nG_00": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "m1.lab.sc_nG_01": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "m1.lab.sc_nG_02": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "m1.rad.sc_J_00": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.rad.sc_J_01": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.rad.sc_J_02": _update_defaults(cmap='plasma', norm='log', vmin=1e-14),
    "m1.rad.sc_n_00": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "m1.rad.sc_n_01": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "m1.rad.sc_n_02": _update_defaults(cmap='viridis', norm='log', vmin=1e45),
    "default": _update_defaults(cmap='viridis', norm='lin'),
}

var_alias: dict[str, str] = {
    "rho": "hydro.prim.rho",
    "p": "hydro.prim.p",
    "ye": "passive_scalar.r_0",
    "s": "hydro.aux.s",
    "m1_E_e": "m1.lab.sc_E_00",
    "m1_E_a": "m1.lab.sc_E_01",
    "m1_E_x": "m1.lab.sc_E_02",
    "m1_nG_e": "m1.lab.sc_nG_00",
    "m1_nG_a": "m1.lab.sc_nG_01",
    "m1_nG_x": "m1.lab.sc_nG_02",
    "m1_J_e": "m1.rad.sc_J_00",
    "m1_J_a": "m1.rad.sc_J_01",
    "m1_J_x": "m1.rad.sc_J_02",
    "m1_n_e": "m1.rad.sc_n_00",
    "m1_n_a": "m1.rad.sc_n_01",
    "m1_n_x": "m1.rad.sc_n_02",
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
    else:
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
    return kwargs
