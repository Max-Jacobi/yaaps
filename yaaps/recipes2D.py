import numpy as np

def det(gxx, gxy, gxz, gyy, gyz, gzz):
     return (-(gxz ** 2) * gyy
             + 2 * gxy * gxz * gyz
             - gxx * (gyz ** 2)
             - (gxy ** 2) * gzz
             + gxx * gyy * gzz)

def gup(gxx, gxy, gxz, gyy, gyz, gzz):
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
    cc = np.zeros((3, len(xyz[0]), len(xyz[0][0]), len(xyz[1][0])))
    i1, i2 = int(samp[0][1])-1, int(samp[1][1])-1
    for imb, coords in enumerate(zip(*xyz)):
        cc[i1][imb], cc[i2][imb] = np.meshgrid(*coords, indexing='ij')
    return cc

def raise_lower(vx, vy, vz,
                gxx, gxy, gxz,
                gyy, gyz, gzz):
    vtx = vx*gxx + vy*gxy + vz*gxz
    vty = vx*gxy + vy*gyy + vz*gyz
    vtz = vx*gxz + vy*gyz + vz*gzz
    return vtx, vty, vtz


def radial_proj(vdx, vdy, vdz, *gd, xyz, sampling):
    xd, yd, zd = untangle_xyz(xyz, sampling)
    xu, yu, zu = raise_lower(xd, yd, zd, *gup(*gd))
    r = np.sqrt(xu*xd + yu*yd + zu*zd)
    return (xu*vdx + yu*vdy + zu*vdz)/r
