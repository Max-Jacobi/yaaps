import numpy as np
from yaaps import Simulation
from yaaps.datatypes import Native, Derived
from yaaps.recipes2D import untangle_xyz, raise_lower

_def_b_lower_dep = ("field.aux.b_u_1", "field.aux.b_u_2", "field.aux.b_u_3",
                  "geom.adm.gxx", "geom.adm.gxy", "geom.adm.gxz",
                  "geom.adm.gyy", "geom.adm.gyz", "geom.adm.gzz")

#utilde = W*v
_def_v_up_dep = ("hydro.prim.util_u_1", "hydro.prim.util_u_2", "hydro.prim.util_u_3", "hydro.aux.W")

def _v_u_r(ux, uy, uz, W, xyz, sampling):
    x, y, z = untangle_xyz(xyz, sampling)
    r_cyl = np.sqrt(x**2 + y**2)
    vx = ux/W
    vy = uy/W
    vz = uz/W
    return (x*vx + y*vy)/r_cyl

def _v_u_phi(ux, uy, uz, W, xyz, sampling):
    x, y, z = untangle_xyz(xyz, sampling)
    r_cyl = np.sqrt(x**2 + y**2)
    vx = ux/W
    vy = uy/W
    vz = uz/W
    return x*vy - y*vx/r_cyl**2

def _b_r(bx_u, by_u, bz_u, *gd, xyz, sampling):
    x, y, z = untangle_xyz(xyz, sampling)
    bx_d, by_d, bz_d = raise_lower(bx_u, by_u, bz_u, *gd)
    r_cyl = np.sqrt(x**2 + y**2)
    return (x*bx_d + y*by_d)/r_cyl

def _b_phi(bx_u, by_u, bz_u, *gd, xyz, sampling):
    x, y, z = untangle_xyz(xyz, sampling)
    bx_d, by_d, bz_d = raise_lower(bx_u, by_u, bz_u, *gd)
    return x*by_d - y*bx_d


sim = Simulation("path/to/simulation")

sampling = ('x1v', 'x2v') # ('x1v', 'x3v')

b_r = Derived(sim, "b_r", _def_b_lower_dep, _b_r, sampling=sampling)
b_phi = Derived(sim, "b_phi", _def_b_lower_dep, _b_phi, sampling=sampling)
v_u_r = Derived(sim, "v_u_r", _def_v_up_dep, _v_u_r, sampling=sampling)
v_u_phi = Derived(sim, "v_u_phi", _def_v_up_dep, _v_u_phi, sampling=sampling)
rho = Native(sim, "hydro.prim.rho", sampling=sampling)
dens = Native(sim, "hydro.cons.D", sampling=sampling)
pres = Native(sim, "hydro.prim.p", sampling=sampling)


def load_data(time):
    (x, y, z), b_r_data, time_br = b_r.load_data(time, strip_ghosts=True)
    (x, y, z), b_phi_data, time_bphi = b_phi.load_data(time, strip_ghosts=True)
    (x, y, z), v_u_r_data, time_vur = v_u_r.load_data(time, strip_ghosts=True)
    (x, y, z), v_u_phi_data, time_vuphi = v_u_phi.load_data(time, strip_ghosts=True)
    (x, y, z), dens_data, time_dens = dens.load_data(time, strip_ghosts=True)
    (x, y, z), rho_data, time_rho = rho.load_data(time, strip_ghosts=True)
    (x, y, z), pres_data, time_pres = pres.load_data(time, strip_ghosts=True)

    assert time_br == time_bphi == time_dens == time_pres == time_rho == time_vur == time_vuphi, "Data loaded at different times"

    dx = x[:, 1] - x[:, 0] # shape (n_mb,)
    dA = (dx**2)[:, None, None]
    weight = dens_data * dA

    return b_r_data, b_phi_data, v_u_r_data, v_u_phi_data, rho_data, pres_data, weight

# x.shape = (n_mb, nx)
# y.shape = (n_mb, ny)
# z.shape = (n_mb, 1)
# data.shape = (n_mb, nx, ny)
