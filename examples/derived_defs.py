import numpy as np
from yaaps.plot2D import DerivedColorPlot
from yaaps.recipes2D import untangle_xyz

def _v_jacobian_2d(vx, vy, x, y):
    #axis 0 is always the meshblock index
    dx = (x[:, 1] - x[:, 0])[:, None, None]
    dy = (y[:, 1] - y[:, 0])[:, None, None]

    dvxdx = np.gradient(vx, axis=1, edge_order=2) / dx
    dvxdy = np.gradient(vx, axis=2, edge_order=2) / dy
    dvydx = np.gradient(vy, axis=1, edge_order=2) / dx
    dvydy = np.gradient(vy, axis=2, edge_order=2) / dy
    return dvxdx, dvxdy, dvydx, dvydy


def _sigmamax_xy(ux, uy, alpha, betax, betay, xyz, sampling):
    # calculate the maximum singular value of the jacobian dv(xy)/d(xy)
    if sampling != ('x1v', 'x2v'):
        raise ValueError(f"Unsupported sampling: {sampling}")
    x, y = xyz
    vx = alpha * ux - betax
    vy = alpha * uy - betay
    dvxdx, dvxdy, dvydx, dvydy = _v_jacobian_2d(-vx, -vy, x, y)

    T = dvxdx**2 + dvxdy**2 + dvydx**2 + dvydy**2
    D = (dvxdx * dvydy - dvxdy * dvydx)**2
    sigmamax =  np.sqrt(0.5 * (T + np.sqrt(T*T - 4*D)))
    return sigmamax

_sigmamax_xy_dep = ("util_u_1", "util_u_2", "adm.alpha", "adm.betax", "adm.betay")

sigmamax_xy = lambda sim, **kwargs: DerivedColorPlot(sim, "sigmamax_xy", _sigmamax_xy_dep, _sigmamax_xy, **kwargs)


def _euler_max_dt_xy(ux, uy, alpha, betax, betay, xyz, sampling):
    # calculate the largest dt that satisfies the Euler stability condition
    if sampling != ('x1v', 'x2v'):
        raise ValueError(f"Unsupported sampling: {sampling}")
    x, y = xyz
    vx = alpha * ux - betax
    vy = alpha * uy - betay
    dvxdx, dvxdy, dvydx, dvydy = _v_jacobian_2d(-vx, -vy, x, y)
    # calculate both (complex) eigenvalues of the jacobian
    trace = dvxdx + dvydy
    det = dvxdx * dvydy - dvxdy * dvydx
    discriminant = trace**2 - 4*det
    sqrt_disc = np.sqrt(discriminant + 0j)
    lambda1 = 0.5 * (trace + sqrt_disc)
    lambda2 = 0.5 * (trace - sqrt_disc)
    abs_lambda1 = np.sqrt(lambda1.real**2 + lambda1.imag**2)
    abs_lambda2 = np.sqrt(lambda2.real**2 + lambda2.imag**2)
    max_real_part = np.maximum(lambda1.real, lambda2.real)
    max_dt = np.zeros_like(max_real_part)
    msk = max_real_part < 0
    max_dt[msk] = np.maximum(
        -2 * lambda1[msk].real / abs_lambda1[msk]**2,
        -2 * lambda2[msk].real / abs_lambda2[msk]
    )
    return max_dt
_euler_max_dt_xy_dep = ("util_u_1", "util_u_2", "adm.alpha", "adm.betax", "adm.betay")
euler_max_dt_xy = lambda sim, **kwargs: DerivedColorPlot(sim, "euler_max_dt_xy", _euler_max_dt_xy_dep, _euler_max_dt_xy, **kwargs)


vars = dict(
    sigmamax_xy = sigmamax_xy,
    euler_max_dt_xy = euler_max_dt_xy,
)
