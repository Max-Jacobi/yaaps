from typing import Callable

from matplotlib.colors import LogNorm, Normalize, AsinhNorm



def _update_defaults(**default) -> Callable[[dict], dict]:
    def _inner(kwargs: dict) -> dict:
        return {**default, **kwargs}
    return _inner

_color_kwargs_default: dict[str, Callable[[dict], dict]] = dict(
    rho=_update_defaults(cmap='magma', norm='log'),
    r0=_update_defaults(cmap='coolwarm_r', norm='lin'), # r0 = ye
    default=_update_defaults(cmap='viridis', norm='lin'),
)

def update_color_kwargs(var: str, kwargs: dict, vmin: float, vmax: float) -> dict:
    if var in _color_kwargs_default:
        kwargs = _color_kwargs_default[var](kwargs)
    else: # default
        kwargs = _color_kwargs_default['default'](kwargs)

    if isinstance(kwargs['norm'], str):
        norm = kwargs.pop('norm', 'lin')
        vmin = kwargs.pop('vmin', vmin)
        vmax = kwargs.pop('vmax', vmax)

        if norm == 'log':
            kwargs['norm'] = LogNorm(vmin=vmin, vmax=vmax)
        elif norm == 'lin':
            kwargs['norm'] = Normalize(vmin=vmin, vmax=vmax)
        elif norm == 'asinh':
            if 'linear_width' not in kwargs:
                raise ValueError('norm=asinh norm requires linear_width argument')
            lin_width = kwargs.pop('linear_width')
            kwargs['norm'] = AsinhNorm(linear_width=lin_width, vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f'Unknown norm: {norm}')
    return kwargs
