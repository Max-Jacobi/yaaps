"""
2D data processing recipes for GRAthena++ simulations.

This module provides mathematical utility functions for working with
metric tensors and coordinate transformations commonly needed in
general relativistic hydrodynamics simulations.
"""

import numpy as np


def det(gxx, gxy, gxz, gyy, gyz, gzz):
    """
    Compute the determinant of a 3x3 symmetric metric tensor.

    Args:
        gxx: The (x,x) component of the metric tensor.
        gxy: The (x,y) component of the metric tensor.
        gxz: The (x,z) component of the metric tensor.
        gyy: The (y,y) component of the metric tensor.
        gyz: The (y,z) component of the metric tensor.
        gzz: The (z,z) component of the metric tensor.

    Returns:
        The determinant of the 3x3 symmetric metric tensor.
    """
    return -(gxz**2) * gyy + 2 * gxy * gxz * gyz - gxx * (gyz**2) - (gxy**2) * gzz + gxx * gyy * gzz


def ye_equilibrium(
    ye,
    eta_nue,
    kap_nue,
    n_nue,
    eta_nua,
    kap_nua,
    n_nua,
    *gd,
    kap_eff_min=1e-6,
):
    """
    Local weak-equilibrium electron fraction from the M1 radiation fields.

    The GR-Athena++ M1 lepton-number source (m1_sources.cpp, m1_utils.hpp
    sources_sc_nG) is

        D dY_e/dt = m_b [S_n(nuebar) - S_n(nue)],
        S_n(s) = sqrt(gamma) eta_0^s - kap_a_0^s n_s,

    so Y_e is raised by nue absorption on neutrons and positron capture on
    neutrons, and lowered by nuebar absorption on protons and electron capture
    on protons. With the weakrates rates (weak_emission.cpp, weak_opacity.cpp)
    each group scales with the free nucleon fractions,

        kap_a_0^nue, eta_0^nuebar ~ eta_np ~ (1 - Y_e),
        kap_a_0^nuebar, eta_0^nue ~ eta_pn ~ Y_e,

    hence, with P = kap_a_0^nue n_nue + sqrt(gamma) eta_0^nuebar = p (1 - Y_e)
    and M = kap_a_0^nuebar n_nuebar + sqrt(gamma) eta_0^nue = m Y_e, the fixed
    point p (1 - Y_e^eq) = m Y_e^eq gives

        Y_e^eq = P Y_e / [P Y_e + M (1 - Y_e)].

    In the optically thin, emission-free limit this reduces to the familiar
    Y_e^eq = 1 / (1 + L_nuebar eps_nuebar / (L_nue eps_nue)) (Qian & Woosley
    1996, eq. 77; Foucart 2024, arXiv:2410.03646, eq. 11).

    Note that M1.rad.sc_n_* is the densitized fluid-frame number density
    (sqrt(gamma) n), which is why only the emissivities carry sqrt(gamma).

    Args:
        ye: Electron fraction (passive_scalar.r_0).
        eta_nue: nue number emissivity (M1.radmat.sc_eta_0_00).
        kap_nue: nue number absorption opacity (M1.radmat.sc_kap_a_0_00).
        n_nue: densitized nue number density (M1.rad.sc_n_00).
        eta_nua: nuebar number emissivity (M1.radmat.sc_eta_0_01).
        kap_nua: nuebar number absorption opacity (M1.radmat.sc_kap_a_0_01).
        n_nua: densitized nuebar number density (M1.rad.sc_n_01).
        *gd: The six components of the covariant 3-metric
            (gxx, gxy, gxz, gyy, gyz, gzz), for sqrt(gamma).
        kap_eff_min: Mask threshold on the effective interaction opacity
            (P + M) / (n_nue + n_nuebar), in code units. Cells below it are
            NaN. This removes (i) the atmosphere, where the weakrates tables
            return exactly zero below rho_min/temp_min, and (ii) the cold,
            dense, strongly degenerate remnant core, where Pauli blocking
            suppresses emission and absorption by ~20 orders of magnitude
            and the balance degenerates into numerical noise. Set to 0 to
            disable the mask.

    Returns:
        The equilibrium electron fraction, NaN where undetermined.
    """
    sqrt_g = np.sqrt(np.abs(det(*gd)))
    plus = kap_nue * n_nue + sqrt_g * eta_nua  # raises Y_e, ~ (1 - Y_e)
    minus = kap_nua * n_nua + sqrt_g * eta_nue  # lowers Y_e, ~ Y_e
    den = plus * ye + minus * (1 - ye)
    ok = (den > 0) & (plus + minus > kap_eff_min * (n_nue + n_nua))
    return np.where(ok, plus * ye / np.where(ok, den, 1), np.nan)


def gup(gxx, gxy, gxz, gyy, gyz, gzz):
    """
    Compute the inverse (contravariant) components of a 3x3 symmetric metric tensor.

    Args:
        gxx: The (x,x) component of the covariant metric tensor.
        gxy: The (x,y) component of the covariant metric tensor.
        gxz: The (x,z) component of the covariant metric tensor.
        gyy: The (y,y) component of the covariant metric tensor.
        gyz: The (y,z) component of the covariant metric tensor.
        gzz: The (z,z) component of the covariant metric tensor.

    Returns:
        A tuple (guxx, guxy, guxz, guyy, guyz, guzz) containing the
        contravariant (inverse) metric tensor components.
    """
    det_g = det(gxx, gxy, gxz, gyy, gyz, gzz)
    oo_det_g = 1 / det_g
    guxx = oo_det_g * (-gyz * gyz + gyy * gzz)
    guxy = oo_det_g * (gxz * gyz - gxy * gzz)
    guxz = oo_det_g * (-gxz * gyy + gxy * gyz)
    guyy = oo_det_g * (-gxz * gxz + gxx * gzz)
    guyz = oo_det_g * (gxy * gxz - gxx * gyz)
    guzz = oo_det_g * (-gxy * gxy + gxx * gyy)
    return guxx, guxy, guxz, guyy, guyz, guzz


def untangle_xyz(xyz, samp):
    """
    Untangle coordinate arrays from meshblock format into a regular 3D grid.

    This function converts coordinate data organized by meshblocks into
    a uniform array suitable for further processing.

    Args:
        xyz: Tuple of coordinate arrays organized by meshblocks.
        samp: Sampling specification tuple, e.g., ('x1v', 'x2v'), used to
            determine which coordinate indices to use.

    Returns:
        A 3D NumPy array of shape (3, n_meshblocks, nx1, nx2) containing
        the meshgrid coordinates for each spatial direction.
    """
    cc = np.zeros((3, len(xyz[0]), len(xyz[0][0]), len(xyz[1][0])))
    i1, i2 = int(samp[0][1]) - 1, int(samp[1][1]) - 1
    for imb, coords in enumerate(zip(*xyz, strict=False)):
        cc[i1][imb], cc[i2][imb] = np.meshgrid(*coords, indexing="ij")
    return cc


def raise_lower(vx, vy, vz, gxx, gxy, gxz, gyy, gyz, gzz):
    """
    Raise or lower vector indices using the metric tensor.

    This function contracts a vector with a metric tensor to convert
    between covariant and contravariant components. The operation
    performed is: v_i = g_ij * v^j (lowering) or v^i = g^ij * v_j (raising),
    depending on whether the input metric is covariant or contravariant.

    Args:
        vx: The x-component of the input vector.
        vy: The y-component of the input vector.
        vz: The z-component of the input vector.
        gxx: The (x,x) component of the metric tensor.
        gxy: The (x,y) component of the metric tensor.
        gxz: The (x,z) component of the metric tensor.
        gyy: The (y,y) component of the metric tensor.
        gyz: The (y,z) component of the metric tensor.
        gzz: The (z,z) component of the metric tensor.

    Returns:
        A tuple (vtx, vty, vtz) containing the transformed vector components.
    """
    vtx = vx * gxx + vy * gxy + vz * gxz
    vty = vx * gxy + vy * gyy + vz * gyz
    vtz = vx * gxz + vy * gyz + vz * gzz
    return vtx, vty, vtz


def radial_proj(vdx, vdy, vdz, *gd, xyz, sampling):
    """
    compute the radial projection of a covariant vector.

    projects a covariant vector onto the radial direction using the
    metric tensor to compute the proper radial distance.

    args:
        vdx: the x-component of the covariant vector.
        vdy: the y-component of the covariant vector.
        vdz: the z-component of the covariant vector.
        *gd: the six independent components of the covariant metric tensor
            (gxx, gxy, gxz, gyy, gyz, gzz).
        xyz: tuple of coordinate arrays organized by meshblocks.
        sampling: sampling specification tuple, e.g., ('x1v', 'x2v').

    returns:
        the radial projection of the vector, computed as (x^i * v_i) / r,
        where r is the proper radial distance.
    """
    xd, yd, zd = untangle_xyz(xyz, sampling)
    xu, yu, zu = raise_lower(xd, yd, zd, *gup(*gd))
    r = np.sqrt(xu * xd + yu * yd + zu * zd)
    return (xu * vdx + yu * vdy + zu * vdz) / r


def normalize_vec(vx, vy, vz, *gd):
    """
    Normalize a covariant or contravariant vector using the metric tensor.
    Computes the norm of a covariant vector and normalizes it to unit length
    using the metric tensor.
    Note that the metric tensor components must match the type of vector
    provided (covariant or contravariant for contravariant or covariant vectors,
    respectively).
    args:
        vdx: the x-component of the covariant vector.
        vdy: the y-component of the covariant vector.
        vdz: the z-component of the covariant vector.
        *gd: the six independent components of the covariant metric tensor
            (gxx, gxy, gxz, gyy, gyz, gzz).
    returns:
        a tuple (nvx, nvy, nvz) containing the normalized vector components.
    """

    vdx, vdy, vdz = raise_lower(vx, vy, vz, *gd)
    norm = np.sqrt(vx * vdx + vy * vdy + vz * vdz)
    nvx = vdx / norm
    nvy = vdy / norm
    nvz = vdz / norm
    return nvx, nvy, nvz


def absolute_val(vx, vy, vz, *gd):
    """
    Compute the absolute value (norm) of a covariant or contravariant vector.

    Computes the norm of a covariant vector using the metric tensor.
    Note that the metric tensor components must match the type of vector
    provided (covariant or contravariant for contravariant or covariant vectors,
    respectively).

    Args:
        vx: The x-component of the covariant vector.
        vy: The y-component of the covariant vector.
        vz: The z-component of the covariant vector.
        *gd: The six independent components of the covariant metric tensor
            (gxx, gxy, gxz, gyy, gyz, gzz).

    Returns:
        The absolute value (norm) of the vector.
    """
    vdx, vdy, vdz = raise_lower(vx, vy, vz, *gd)
    return np.sqrt(vx * vdx + vy * vdy + vz * vdz)
