import numpy as np

from yaaps.recipes2D import ye_equilibrium

FLAT = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)  # gxx, gxy, gxz, gyy, gyz, gzz


def test_balanced_rates_are_already_in_equilibrium():
    ye = np.array([0.1, 0.3])
    # kap_nue * n_nue == kap_nua * n_nua and eta_nue == eta_nua => P == M,
    # so the fixed point must be Y_e itself.
    got = ye_equilibrium(ye, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, *FLAT)
    assert np.allclose(got, ye)


def test_pure_nue_absorption_drives_ye_to_one():
    got = ye_equilibrium(np.array([0.1]), 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *FLAT)
    assert np.allclose(got, 1.0)


def test_blocked_and_empty_cells_are_masked():
    # all rates zero (atmosphere) and rates far below the floor (blocked core)
    got = ye_equilibrium(
        np.array([0.1, 0.1]),
        np.array([0.0, 0.0]),
        np.array([0.0, 1e-30]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 1e-30]),
        np.array([0.0, 1.0]),
        *FLAT,
    )
    assert np.all(np.isnan(got))
