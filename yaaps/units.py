import re

units: dict[str | re.Pattern, tuple[float, str]] = {
    "rho": (6.175828477586656e+17, " [g cm$^{-3}$]"),
    "eps": (8.9875517873681764e+20, " [erg g$^{-1}$]"),
    "P": (5.550725674743868e+38, " [erg cm$^{-3}$]"),
    # "mass": (1.988409870967742e+33, " [g]"),
    "energy": (1.7870936689836656e+53, " [erg]"),
    "time": (0.004925490948309319, " [ms]"),
    # "r": (1.4766250382504018, " [km]"),
    # "x": (1.4766250382504018, " [km]"),
    # "y": (1.4766250382504018, " [km]"),
    # "z": (1.4766250382504018, " [km]"),
    "mass": (1.0, r" [$M_\odot$]"),
    re.compile(r"nu\d_lum"): (3.628132869648639e+59, " [erg s$^{-1}$]"),
    re.compile(r"nu\d_en"): (1.11545707207968e+60, " [MeV]"),
    re.compile("m_ej"): (1.0, r" [$M_\odot$]"),
    re.compile("mdot_ej"): (203025.44670054692, r" [$M_\odot$ s$^{-1}$]"),
    re.compile("util_u"): (1.0, r" [$c$]"),
    re.compile("vel"): (1.0, r" [$c$]"),
    }

def apply_units(key: str) -> tuple[float, str]:
    for uk in units:
        if isinstance(uk, str) and key.endswith(uk):
            return units[uk]
        if isinstance(uk, re.Pattern) and re.match(uk, key) is not None:
            return units[uk]
    return 1.0, ""
