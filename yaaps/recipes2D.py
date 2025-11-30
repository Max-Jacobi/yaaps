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
    return (-(gxz ** 2) * gyy
            + 2 * gxy * gxz * gyz
            - gxx * (gyz ** 2)
            - (gxy ** 2) * gzz
            + gxx * gyy * gzz)


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
    oo_det_g = 1/det_g
    guxx = oo_det_g * (-gyz*gyz + gyy*gzz)
    guxy = oo_det_g * ( gxz*gyz - gxy*gzz)
    guxz = oo_det_g * (-gxz*gyy + gxy*gyz)
    guyy = oo_det_g * (-gxz*gxz + gxx*gzz)
    guyz = oo_det_g * ( gxy*gxz - gxx*gyz)
    guzz = oo_det_g * (-gxy*gxy + gxx*gyy)
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
    i1, i2 = int(samp[0][1])-1, int(samp[1][1])-1
    for imb, coords in enumerate(zip(*xyz)):
        cc[i1][imb], cc[i2][imb] = np.meshgrid(*coords, indexing='ij')
    return cc


def raise_lower(vx, vy, vz,
                gxx, gxy, gxz,
                gyy, gyz, gzz):
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
    vtx = vx*gxx + vy*gxy + vz*gxz
    vty = vx*gxy + vy*gyy + vz*gyz
    vtz = vx*gxz + vy*gyz + vz*gzz
    return vtx, vty, vtz


def radial_proj(vdx, vdy, vdz, *gd, xyz, sampling):
    """
    Compute the radial projection of a covariant vector.

    Projects a covariant vector onto the radial direction using the
    metric tensor to compute the proper radial distance.

    Args:
        vdx: The x-component of the covariant vector.
        vdy: The y-component of the covariant vector.
        vdz: The z-component of the covariant vector.
        *gd: The six independent components of the covariant metric tensor
            (gxx, gxy, gxz, gyy, gyz, gzz).
        xyz: Tuple of coordinate arrays organized by meshblocks.
        sampling: Sampling specification tuple, e.g., ('x1v', 'x2v').

    Returns:
        The radial projection of the vector, computed as (x^i * v_i) / r,
        where r is the proper radial distance.
    """
    xd, yd, zd = untangle_xyz(xyz, sampling)
    xu, yu, zu = raise_lower(xd, yd, zd, *gup(*gd))
    r = np.sqrt(xu*xd + yu*yd + zu*zd)
    return (xu*vdx + yu*vdy + zu*vdz)/r
