from multiprocessing import Pool
from sys import stdout
from typing import Callable, Iterable, Sized, Any
from functools import partial
import numpy as np
from tqdm import tqdm


################################################################################
# parallele helpers

def _apply_tail(func: Callable[..., Any], tail: tuple[Any, ...], item: Any):
    return func(item, *tail)

def do_parallel[R](
    func: Callable[..., R],
    itr: Iterable,
    n_cpu: int,
    args: tuple[Any, ...] = (),
    verbose: bool = False,
    ordered: bool = False,
    **kwargs
) -> Iterable[R]:
    if isinstance(itr, Sized):
        kwargs["total"] = len(itr)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    func_args = partial(_apply_tail, func, args)

    if n_cpu == 1:
        return tqdm(map(func_args, itr), **kwargs)
    pool = Pool(n_cpu)
    if ordered:
        return tqdm(pool.imap(func_args, itr), **kwargs)
    return tqdm(pool.imap_unordered(func_args, itr), **kwargs)

def _index_then_apply[T, R](ix_x: tuple[int, T], func: Callable[..., R], tail: tuple[Any, ...]) -> tuple[int, R]:
    i, x = ix_x
    return i, func(x, *tail)

def do_parallel_enumerate[T, R](
    func: Callable[..., R],
    itr: Iterable[T],
    n_cpu: int,
    args: tuple[Any, ...] = (),
    verbose: bool = False,
    **kwargs
) -> Iterable[tuple[int, R]]:
    return do_parallel(
        _index_then_apply,
        enumerate(itr),
        n_cpu=n_cpu,
        args=(func, args),
        verbose=verbose,
        **kwargs,
    )

################################################################################
def det(g):
     gxx, gxy, gxz, gyy, gyz, gzz = g
     return (-(gxz ** 2) * gyy
             + 2 * gxy * gxz * gyz
             - gxx * (gyz ** 2)
             - (gxy ** 2) * gzz
             + gxx * gyy * gzz)

def gup(gd):
    det_g = det(gd)
    gxx, gxy, gxz, gyy, gyz, gzz = gd
    oo_det_g = 1/det_g
    guxx = oo_det_g * (-gyz*gyz + gyy*gzz)
    guxy = oo_det_g * ( gxz*gyz - gxy*gzz)
    guxz = oo_det_g * (-gxz*gyy + gxy*gyz)
    guyy = oo_det_g * (-gxz*gxz + gxx*gzz)
    guyz = oo_det_g * ( gxy*gxz - gxx*gyz)
    guzz = oo_det_g * (-gxy*gxy + gxx*gyy)
    return guxx, guxy, guxz, guyy, guyz, guzz

def raise_lower(v, g):
    gxx, gxy, gxz, gyy, gyz, gzz = g
    vx, vy, vz = v
    vtx = vx*gxx + vy*gxy + vz*gxz
    vty = vx*gxy + vy*gyy + vz*gyz
    vtz = vx*gxz + vy*gyz + vz*gzz
    return vtx, vty, vtz

def v_abs(cd, gu):
    cu = raise_lower(cd, gu)
    return np.sqrt(sum(u*d for u, d in zip(cu, cd)))

def radial_proj(vu, gd, cd):
    r = v_abs(cd, gup(gd))
    return sum(c*v for c, v in zip(vu, cd))/r

def v_adv_u(util_u, alp, bet_u):
    return tuple(u*alp - b for u, b in zip(util_u, bet_u))

def m_flux_ej(D_til, util_u_x, util_u_y, util_u_z,
              alp, bet_u_x, bet_u_y, bet_u_z, hu_t,
              g_d_xx, g_d_xy, g_d_xz,
              g_d_yy, g_d_yz, g_d_zz,
              xd, yd, zd):
    gd = (g_d_xx, g_d_xy, g_d_xz, g_d_yy, g_d_yz, g_d_zz)
    cd = (xd, yd, zd)
    uu = (util_u_x, util_u_y, util_u_z)
    bu = (bet_u_x, bet_u_y, bet_u_z,)
    D_ej = D_til.copy()
    D_ej[hu_t > -1] = 0
    va_u = v_adv_u(uu, alp, bu)
    F_ej_u = tuple(D_ej*v for v in va_u)
    return radial_proj(F_ej_u, gd, cd)
