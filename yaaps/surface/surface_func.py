import numpy as np
from typing import Callable, Literal, Iterable, Final
from operator import truediv

################################################################################

Scalar = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Vector_u = np.ndarray[tuple[Literal[3], int, int], np.dtype[np.float64]]
Vector_d = np.ndarray[tuple[Literal[3], int, int], np.dtype[np.float64]]
AnyField = (Scalar | Vector_d | Vector_u)
Histogram = tuple[np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]

class SurfaceFunc[R]:
    def __init__(
        self,
        func: Callable[..., R],
        keys: Iterable[str],
        name: str,
        **kwargs
        ):
        self._func: Final[Callable[..., R]] = func
        self.keys = list(keys)
        self.name = name
        self.kwargs = kwargs

    def __call__(self, *args) -> R:
        return self._func(*args, **self.kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"

def _id(arg: AnyField) -> AnyField:
    return arg
def get_func(func: (str | SurfaceFunc[AnyField])) -> SurfaceFunc[AnyField]:
    if isinstance(func, str):
        return SurfaceFunc(_id, (func, ), func)
    return func



class DerivedSurfaceFunc[R](SurfaceFunc):
    def __init__(
        self,
        funcs: Iterable[SurfaceFunc | str],
        func: Callable[..., R],
        name: str,
        **kwargs
        ):
        self.dependence = tuple(funcs)

        keys, bare_funcs, derived_funcs = set(), set(), set()

        for f in map(get_func, funcs):
            DerivedSurfaceFunc._sort_func(f, keys, derived_funcs, bare_funcs)

        self.bare_funcs = tuple(bare_funcs)
        self.derived_funcs = tuple(derived_funcs)
        super().__init__(func, keys, name, **kwargs)

        self.bare_idx = tuple((b, tuple(self.keys.index(k) for k in b.keys))
                             for b in self.bare_funcs)
        self.idx = tuple((f, self.bare_funcs.index(f), b) if (b := f in self.bare_funcs) else
                         (f, self.derived_funcs.index(f), b) for f in self.dependence)

    def __call__(self, *args) -> R:
        res = {b: b(*(args[i] for i in ik)) for b, ik in self.bare_idx}
        def _get_args(f):
            a = []
            for d in f.dependence:
                if d in res:
                    a.append(res[d])
                else:
                    a.append(d._func(*_get_args(d), **d.kwargs))
            return a
        return self._func(*_get_args(self), **self.kwargs)

    @staticmethod
    def _add_dependence(f: SurfaceFunc, keys: set, dependence: set):
        dependence.add(f)
        for k in f.keys:
            keys.add(k)

    @staticmethod
    def _sort_func(f: SurfaceFunc, keys: set, derived: set, bare: set):
        if type(f) is SurfaceFunc:
            DerivedSurfaceFunc._add_dependence(f, keys, bare)
        elif isinstance(f, DerivedSurfaceFunc):
            derived.add(f)
            for d in f.bare_funcs:
                DerivedSurfaceFunc._add_dependence(d, keys, bare)
        else:
            raise ValueError(f"Unknown SurfaceFunc type {f.__class__} for func {f.name}")


################################################################################

def _crit(ut: Scalar) -> Scalar:
    return (ut < -1).astype(float)

def _scalar_prod(x: Vector_u, y: Vector_d) -> Scalar:
    return (x*y).sum(axis=0)

def _det(gxx, gxy, gxz, gyy, gyz, gzz):
     return (-(gxz ** 2) * gyy
             + 2 * gxy * gxz * gyz
             - gxx * (gyz ** 2)
             - (gxy ** 2) * gzz
             + gxx * gyy * gzz)

def _gup(gxx, gxy, gxz, gyy, gyz, gzz):
    det_g = _det(gxx, gxy, gxz, gyy, gyz, gzz)
    oo_det_g = 1/det_g
    guxx = oo_det_g * (-gyz*gyz + gyy*gzz)
    guxy = oo_det_g * ( gxz*gyz - gxy*gzz)
    guxz = oo_det_g * (-gxz*gyy + gxy*gyz)
    guyy = oo_det_g * (-gxz*gxz + gxx*gzz)
    guyz = oo_det_g * ( gxy*gxz - gxx*gyz)
    guzz = oo_det_g * (-gxy*gxy + gxx*gyy)
    return guxx, guxy, guxz, guyy, guyz, guzz

def _raise_lower(v, gxx, gxy, gxz, gyy, gyz, gzz):
    vx, vy, vz = v
    vtx = vx*gxx + vy*gxy + vz*gxz
    vty = vx*gxy + vy*gyy + vz*gyz
    vtz = vx*gxz + vy*gyz + vz*gzz
    return vtx, vty, vtz

def _v_abs(cd, *gd):
    cu = _raise_lower(cd, *_gup(*gd))
    return np.sqrt(sum(u*d for u, d in zip(cu, cd)))

################################################################################


def _nr_d(x, y, z, *gd) -> Vector_d:
    cd = np.array((x, y, z))
    r = _v_abs(cd, *gd)
    return cd/r

nr_d =  SurfaceFunc(
    _nr_d,
    ("x", "y", "z",
     "geom.z4c.gxx", "geom.z4c.gxy", "geom.z4c.gxz",
     "geom.z4c.gyy", "geom.z4c.gyz", "geom.z4c.gzz"),
    "surface normal",
    )

################################################################################


def _dA_d(x, y, z, dA, *gd) -> Vector_d:
    cd = np.array((x, y, z))
    r = _v_abs(cd, *gd)
    return cd/r*dA

dA_d =  SurfaceFunc(
    _dA_d,
    ("x", "y", "z", "dA",
     "geom.z4c.gxx", "geom.z4c.gxy", "geom.z4c.gxz",
     "geom.z4c.gyy", "geom.z4c.gyz", "geom.z4c.gzz"),
    "surf_element",
    )

################################################################################


def _nu_nlum_flux(
    J, st_H_u_t, n, W, alp,
    st_H_u_x, st_H_u_y, st_H_u_z,
    util_u_x, util_u_y, util_u_z,
    bet_u_x, bet_u_y, bet_u_z,
    ) -> Vector_u:
    stHu = np.array((st_H_u_x, st_H_u_y, st_H_u_z,))
    uu = np.array((util_u_x, util_u_y, util_u_z))
    bu = np.array((bet_u_x, bet_u_y, bet_u_z,))

    sp_H_u = st_H_u_t*bu + stHu
    J = np.clip(J, 1e-40, None)
    return n*(alp*uu - W*bu + alp*sp_H_u/J)

def _nu_nlum_keys(inu: int) -> tuple[str, ...]:
    return  (
        f"M1.rad.J_{inu:02d}", f"M1.rad.st_H_u_t_{inu:02d}", f"M1.rad.n_{inu:02d}",
        "hydro.aux.W", "geom.z4c.alpha",
        f"M1.rad.st_H_u_x_{inu:02d}", f"M1.rad.st_H_u_y_{inu:02d}", f"M1.rad.st_H_u_z_{inu:02d}",
        "hydro.prim.util_u_1", "hydro.prim.util_u_2", "hydro.prim.util_u_3",
        "geom.z4c.betax", "geom.z4c.betay", "geom.z4c.betaz",
        )

nu0_nlum_flux = SurfaceFunc(_nu_nlum_flux, _nu_nlum_keys(0), "nu_nlum_00")
nu1_nlum_flux = SurfaceFunc(_nu_nlum_flux, _nu_nlum_keys(1), "nu_nlum_01")
nu2_nlum_flux = SurfaceFunc(_nu_nlum_flux, _nu_nlum_keys(2), "nu_nlum_02")

################################################################################

def _nu_lum_flux(
    F_d_x, F_d_y, F_d_z, en, alp,
    bet_u_x, bet_u_y, bet_u_z, *gd,
    ) -> Vector_u:
    Fd = np.array((F_d_x, F_d_y, F_d_z,))
    bu = np.array((bet_u_x, bet_u_y, bet_u_z,))

    Fu =  _raise_lower(Fd, *_gup(*gd))
    return alp*Fu - en*bu

def _nu_lum_keys(inu: int) -> tuple[str, ...]:
    return  (
        f"M1.lab.F_d_x_{inu:02d}", f"M1.lab.F_d_y_{inu:02d}", f"M1.lab.F_d_z_{inu:02d}",
        f"M1.lab.E_{inu:02d}", "geom.z4c.alpha",
        "geom.z4c.betax", "geom.z4c.betay", "geom.z4c.betaz",
        "geom.z4c.gxx", "geom.z4c.gxy", "geom.z4c.gxz",
        "geom.z4c.gyy", "geom.z4c.gyz", "geom.z4c.gzz",
        )

nu0_lum_flux = SurfaceFunc(_nu_lum_flux, _nu_lum_keys(0), "nu_lum_00")
nu1_lum_flux = SurfaceFunc(_nu_lum_flux, _nu_lum_keys(1), "nu_lum_01")
nu2_lum_flux = SurfaceFunc(_nu_lum_flux, _nu_lum_keys(2), "nu_lum_02")

################################################################################

def _V_u(
    alp, W,
    util_u_x, util_u_y, util_u_z,
    bet_u_x, bet_u_y, bet_u_z,
    ) -> Vector_u:
    uu = np.array((util_u_x, util_u_y, util_u_z))
    bu = np.array((bet_u_x, bet_u_y, bet_u_z))
    return uu/W*alp - bu

V_u =  SurfaceFunc(
    _V_u,
    ("geom.z4c.alpha", "hydro.aux.W",
     "hydro.prim.util_u_1", "hydro.prim.util_u_2", "hydro.prim.util_u_3",
     "geom.z4c.betax", "geom.z4c.betay", "geom.z4c.betaz"),
    "V_u",
    )

def _mass_flux(D_til, *args) -> Vector_u:
    return D_til*_V_u(*args)

mass_flux =  SurfaceFunc(
    _mass_flux,
    ("hydro.cons.D", *V_u.keys),
    "mass_flux",
    )

################################################################################


def radial_projection(vec: (str | SurfaceFunc[Vector_u])) -> DerivedSurfaceFunc[Scalar]:
    _vec = get_func(vec)
    return DerivedSurfaceFunc((_vec, nr_d), _scalar_prod, f"{_vec.name}_r")

def _one() -> Scalar:
    return np.ones((1,1))
no_crit = SurfaceFunc(_one, (), 'no_crit')

def criterion(u_t: (str | SurfaceFunc[AnyField])) -> DerivedSurfaceFunc[Scalar]:
    _u_t = get_func(u_t)
    return DerivedSurfaceFunc((_u_t, ), _crit, f"Crit({_u_t.name})")

################################################################################
def _surf_int(
    v: Vector_u,
    dA: Vector_d,
    crit: Scalar,
    weight: (AnyField | None) = None
    ) -> float:
    f_r = v*dA
    if crit is not None:
        f_r *= crit
    if weight is None:
        return f_r.sum()
    return (f_r*weight).sum()


def _surf_int_out(
    v: Vector_u,
    dA: Vector_d,
    crit: Scalar,
    weight: AnyField | None = None
    ) -> float:
    f_r = np.clip((v*dA).sum(0), 0, None)
    if crit is not None:
        f_r *= crit
    if weight is None:
        return f_r.sum()
    return (f_r*weight).sum()


def integrate_flux(
    flux: (str | SurfaceFunc[Vector_u]),
    weight: (None | str | SurfaceFunc[AnyField]) = None,
    crit: (None | str | SurfaceFunc[Scalar]) = None,
    out: bool = False,
    ) -> DerivedSurfaceFunc[float]:
    c = no_crit if crit is None else criterion(crit)
    surf_int = _surf_int_out if out else _surf_int
    _flux = get_func(flux)
    name = f"Integral({_flux.name})"
    if weight is None:
        return DerivedSurfaceFunc((_flux, dA_d, c), surf_int, name)
    _weight = get_func(weight)
    name = f"Integral({_flux.name}*{_weight.name})"
    return DerivedSurfaceFunc((_flux, dA_d, c, _weight), surf_int, name)

################################################################################

def _hist(
    mass_flux: Vector_u,
    dA: Vector_d,
    func: AnyField,
    crit: Scalar,
    **kwargs
    ) -> Histogram:
    f_r = (mass_flux*dA).sum(0) * crit
    return np.histogram(func.flatten(), weights=f_r.flatten(), **kwargs)

def _hist_out(
    mass_flux: Vector_u,
    dA: Vector_d,
    func: AnyField,
    crit: Scalar,
    **kwargs
    ) -> Histogram:
    f_r = (mass_flux*dA).sum(0) * crit
    f_r = np.clip(f_r, 0, None)
    return np.histogram(func.flatten(), weights=f_r.flatten(), **kwargs)


def mass_histogram(
    func: str | SurfaceFunc[AnyField],
    crit: None | str | SurfaceFunc[Scalar] = None,
    out: bool = False,
    **kwargs
    ) -> DerivedSurfaceFunc[Histogram]:
    c = no_crit if crit is None else criterion(crit)
    hist = _hist_out if out else _hist
    _func = get_func(func)
    name = f"Histogram({_func.name})"
    return DerivedSurfaceFunc((mass_flux, dA_d, _func, c), hist, name, **kwargs)

################################################################################

def _return_tuple[R](*args: R) -> tuple[R, ...]:
    return args

def get_many(funcs: Iterable[SurfaceFunc]):
    return DerivedSurfaceFunc(funcs, _return_tuple, ", ".join(f.name for f in funcs))

################################################################################

def _ut(c: str) -> str:
    if c.startswith('bernoulli'):
        return 'hydro.aux.hu_t'
    if c.startswith('geodesic'):
        return 'hydro.aux.u_t'
    raise ValueError

def _vinf(ut: Scalar) -> Scalar:
    vinf = np.zeros_like(ut)
    mask = ut < -1
    vinf[mask] = np.sqrt(1 - ut[mask]**-2)
    return vinf

vinf = {c: SurfaceFunc(_vinf, (_ut(c),), 'vinf_b') for c in ("bernoulli", "geodesic")}

################################################################################

def wmdot(w: SurfaceFunc[AnyField] | str | None) -> dict[str, SurfaceFunc[float]]:
    return {
        'none': integrate_flux(mass_flux, weight=w),
        'none_out': integrate_flux(mass_flux, weight=w, out=True),
        "bernoulli": integrate_flux(mass_flux, weight=w, crit="hydro.aux.hu_t"),
        "geodesic": integrate_flux(mass_flux, weight=w, crit="hydro.aux.u_t"),
        "bernoulli_out": integrate_flux(mass_flux, weight=w, crit="hydro.aux.hu_t", out=True),
        "geodesic_out": integrate_flux(mass_flux, weight=w, crit="hydro.aux.u_t", out=True),
        }

mdot = wmdot(None)

def hist(w: SurfaceFunc[AnyField] | str, **kwargs) -> dict[str, SurfaceFunc[Histogram]]:
    return {
        'none': mass_histogram(w, **kwargs),
        'none_out': mass_histogram(w, out=True, **kwargs),
        "bernoulli": mass_histogram(w, crit="hydro.aux.hu_t", **kwargs),
        "geodesic": mass_histogram(w, crit="hydro.aux.u_t", **kwargs),
        "bernoulli_out": mass_histogram(w, crit="hydro.aux.hu_t", out=True, **kwargs),
        "geodesic_out": mass_histogram(w, crit="hydro.aux.u_t", out=True, **kwargs),
        }
################################################################################

nu0_lum = integrate_flux(nu0_lum_flux)
nu1_lum = integrate_flux(nu1_lum_flux)
nu2_lum = integrate_flux(nu2_lum_flux)
nu0_nlum = integrate_flux(nu0_nlum_flux)
nu1_nlum = integrate_flux(nu1_nlum_flux)
nu2_nlum = integrate_flux(nu2_nlum_flux)
nu0_e = DerivedSurfaceFunc((nu0_lum, nu0_nlum), truediv, 'e_nu0')
nu1_e = DerivedSurfaceFunc((nu1_lum, nu1_nlum), truediv, 'e_nu1')
nu2_e = DerivedSurfaceFunc((nu2_lum, nu2_nlum), truediv, 'e_nu2')

nu_lum = (nu0_lum, nu1_lum, nu2_lum)
nu_nlum = (nu0_nlum, nu1_nlum, nu2_nlum)
nu_e = (nu0_e, nu1_e, nu2_e)
