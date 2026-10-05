"""
Unit conversion module for GRAthena++ simulations.

This module provides unit conversion factors and labels for converting
simulation quantities from code units to physical (CGS or natural) units.
The conversions are based on typical neutron star merger simulation scales.

This module provides:
- UnitConverter: Class for handling unit conversions from code to physical units
- FieldLabels: Class for providing LaTeX labels for field names
- apply_units(): Backward-compatible function for unit lookups

Example:
    >>> from yaaps.units import UnitConverter, FieldLabels
    >>> converter = UnitConverter()
    >>> scale, unit = converter.get_conversion("rho")
    >>> print(f"Scale: {scale}, Unit: {unit}")
    Scale: 6.175828477586656e+17, Unit:  [g cm$^{-3}$]
    >>> labels = FieldLabels()
    >>> labels.get_label("rho")
    '$\\rho$'
"""

import re

# Dictionary mapping variable names/patterns to (conversion_factor, unit_label) tuples.
# The conversion factor converts from code units to the displayed units.
# Keys can be strings (matched by endswith) or compiled regex patterns.
units: dict[str | re.Pattern, tuple[float, str]] = {
    # Dimensionless .hst columns that would otherwise be claimed by a generic
    # suffix key below ("r" -> km). Patterns are matched with re.match, so
    # these are anchored and cannot leak onto other names.
    re.compile("(max_abs|L1)_Xcons_err$"): (1.0, ""),
    # .hst passive-scalar integrals int D r_N would end in "r" (km): mass-weighted,
    # in M_sun; SCMIX (transition EOS, --nscalars=8) is eps in units of the baryon
    # mass, so its integral is M_sun c^2.
    re.compile(r"7-scalar$"): (1.7870936689836656e54, " [erg]"),
    re.compile(r"\d+-scalar$"): (1.0, r" [$M_\odot$]"),
    # RHINE / transition-EOS derived fields (gr-athena eos_utils.cpp). Listed
    # before the generic suffix keys so e.g. X_err is not claimed by "r" (km).
    # heating_rate and the rhine_d* rates are written in physical units already;
    # qdot_code / fnu_lum are densitized code rates (energy per time per volume).
    "hydro.aux.heating_rate": (1.0, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.qdot_code": (5.550725674743868e38 / 4.925490948309319e-6, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.fnu_lum": (5.550725674743868e38 / 4.925490948309319e-6, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.rhine_qphys": (5.550725674743868e38 / 4.925490948309319e-6, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.rhine_qexit": (5.550725674743868e38 / 4.925490948309319e-6, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.rhine_qreent": (5.550725674743868e38 / 4.925490948309319e-6, r" [erg cm$^{-3}$ s$^{-1}$]"),
    "hydro.aux.nse_state": (1.0, ""),
    "hydro.aux.transition_w": (1.0, ""),
    "hydro.aux.X_err": (1.0, ""),
    "hydro.aux.fnu": (1.0, ""),
    re.compile(r"hydro\.aux\.rhine_d(ye|yn|yp|ya|yh|ah)$"): (1.0, r" [s$^{-1}$]"),
    "hydro.aux.rhine_dma": (1.0, r" [MeV s$^{-1}$]"),
    "rho": (6.175828477586656e17, " [g cm$^{-3}$]"),
    "aux.e": (6.175828477586656e17, " [g cm$^{-3}$]"),
    "aux.T": (1.0, " [MeV]"),
    "aux.s": (1.0, r" [$k_{\mathrm{B}}$]"),
    "eps": (8.9875517873681764e20, " [erg g$^{-1}$]"),
    "P": (5.550725674743868e38, " [erg cm$^{-3}$]"),
    # "mass": (1.988409870967742e+33, " [g]"),
    "energy": (1.7870936689836656e53, " [erg]"),
    "time": (0.004925490948309319, " [ms]"),
    "r": (1.4766250382504018, " [km]"),
    "x": (1.4766250382504018, " [km]"),
    "y": (1.4766250382504018, " [km]"),
    "z": (1.4766250382504018, " [km]"),
    "mass": (1.0, r" [$M_\odot$]"),
    "Omega": (203025.44670054692, " [s$^{-1}$]"),
    re.compile(r"nu\d_lum"): (3.628132869648639e59, " [erg s$^{-1}$]"),
    re.compile(r"nu\d_en"): (1.11545707207968e60, " [MeV]"),
    re.compile("m_ej"): (1.0, r" [$M_\odot$]"),
    re.compile("mdot_ej"): (203025.44670054692, r" [$M_\odot$ s$^{-1}$]"),
    re.compile("util_u"): (1.0, r" [$c$]"),
    re.compile("vel"): (1.0, r" [$c$]"),
    re.compile("x[1-3][vf]?"): (1.4766250382504018, " [km]"),
    # history (.hst) columns. max_rho/min_rho already match "rho" by suffix;
    # the RHINE slots are volume-integrated code-unit rates (energy per time,
    # see gr-athena outputs/history.cpp).
    "max_T": (1.0, " [MeV]"),
    "min_T": (1.0, " [MeV]"),
    "rhine-qdot": (3.628132869648639e59, " [erg s$^{-1}$]"),
    "rhine-Lfnu": (3.628132869648639e59, " [erg s$^{-1}$]"),
    "rhine-qphys": (3.628132869648639e59, " [erg s$^{-1}$]"),
    "rhine-qexit": (3.628132869648639e59, " [erg s$^{-1}$]"),
    "rhine-qreent": (3.628132869648639e59, " [erg s$^{-1}$]"),
    "m-nonNSE": (1.0, r" [$M_\odot$]"),
    "dt": (0.004925490948309319, " [ms]"),
}


class UnitConverter:
    """
    Handles unit conversions from code units to physical units.

    This class provides methods to look up conversion factors and unit strings
    for simulation variables. It supports both exact suffix matching and
    regex pattern matching for flexible variable name lookups.

    Attributes:
        _conversions: Internal dictionary mapping variable names/patterns to
            (scale_factor, unit_string) tuples.
    """

    def __init__(self):
        """
        Initialize the UnitConverter with default conversion factors.

        The default conversions include common simulation variables like
        density (rho), specific energy (eps), pressure (P), time, and
        coordinate conversions.
        """
        # Start with a copy of the global units dict
        self._conversions: dict[str | re.Pattern, tuple[float, str]] = dict(units)

        # Add coordinate conversions
        coord_scale = 1.4766250382504018
        coord_unit = r" [km]"
        for coord in ("x1v", "x2v", "x3v", "x1f", "x2f", "x3f"):
            self._conversions[coord] = (coord_scale, coord_unit)

    def get_conversion(self, field_name: str) -> tuple[float, str]:
        """
        Return the conversion factor and unit string for a field.

        Looks up the variable name in the conversions dictionary.
        Supports both exact suffix matching (for string keys) and
        regex pattern matching.

        Args:
            field_name: The variable name to look up, e.g., "rho", "x1v".

        Returns:
            A tuple (scale_factor, unit_string) where:
            - scale_factor is a float to multiply code values by
            - unit_string is a string suitable for axis labels (with LaTeX)

            Returns (1.0, "") if no matching conversion is found.
        """
        for key in self._conversions:
            if isinstance(key, str) and field_name.endswith(key):
                return self._conversions[key]
            if isinstance(key, re.Pattern) and re.match(key, field_name) is not None:
                return self._conversions[key]
        return 1.0, ""

    def add_unit(self, field_name: str, scale: float, unit: str) -> None:
        """
        Add or update a unit conversion.

        Args:
            field_name: The variable name or pattern to add.
            scale: The conversion scale factor from code to physical units.
            unit: The unit string (with LaTeX formatting if desired).
        """
        self._conversions[field_name] = (scale, unit)


class FieldLabels:
    """
    Provides pretty LaTeX labels for field names.

    This class maps simulation variable names (both short aliases and
    full internal names) to LaTeX-formatted strings for publication-ready
    figures.

    Attributes:
        _labels: Internal dictionary mapping field names to LaTeX strings.
    """

    def __init__(self):
        """
        Initialize FieldLabels with default label mappings.

        The default labels include common coordinates, hydrodynamic
        variables, passive scalars, velocities, and magnetic fields.
        """
        self._labels: dict[str, str] = {
            # Coordinates
            "x1": r"$x$",
            "x2": r"$y$",
            "x3": r"$z$",
            "x1v": r"$x$",
            "x2v": r"$y$",
            "x3v": r"$z$",
            "x1f": r"$x$",
            "x2f": r"$y$",
            "x3f": r"$z$",
            "time": r"$t$",
            # Hydrodynamic variables - short aliases
            "rho": r"$\rho$",
            "p": r"$P$",
            "P": r"$P$",
            "eps": r"$\varepsilon$",
            # Hydrodynamic variables - full names
            "hydro.prim.rho": r"$\rho$",
            "hydro.prim.p": r"$P$",
            "hydro.aux.s": r"$s$",
            "hydro.aux.T": r"$T$",
            "hydro.aux.e": r"$e$",
            # Passive scalars
            "ye": r"$Y_e$",
            "passive_scalar.r_0": r"$Y_e$",
            "s": r"$s$",
            # Velocities - short aliases
            "util_x": r"$\tilde{u}^x$",
            "util_y": r"$\tilde{u}^y$",
            "util_z": r"$\tilde{u}^z$",
            # Velocities - full names
            "hydro.prim.util_u_1": r"$\tilde{u}^x$",
            "hydro.prim.util_u_2": r"$\tilde{u}^y$",
            "hydro.prim.util_u_3": r"$\tilde{u}^z$",
            # Magnetic fields - short aliases
            "B_x": r"$B^x$",
            "B_y": r"$B^y$",
            "B_z": r"$B^z$",
            "b_x": r"$b^x$",
            "b_y": r"$b^y$",
            "b_z": r"$b^z$",
            # Magnetic fields - full names
            "B.Bcc_1": r"$B^x$",
            "B.Bcc_2": r"$B^y$",
            "B.Bcc_3": r"$B^z$",
            "field.aux.b_u_1": r"$b^x$",
            "field.aux.b_u_2": r"$b^y$",
            "field.aux.b_u_3": r"$b^z$",
            # RHINE / transition EOS
            "hydro.aux.heating_rate": r"$\dot{q}_{\mathrm{RHINE}}$",
            "hydro.aux.qdot_code": r"$\alpha\sqrt{\gamma}\,\dot{q}$",
            "hydro.aux.fnu_lum": r"$\alpha\sqrt{\gamma}\,\dot{q}_{\nu}$",
            "hydro.aux.transition_w": r"$w_{\mathrm{NSE}}$",
            "hydro.aux.X_err": r"$\Delta X$",
            "hydro.aux.fnu": r"$f_{\nu}$",
            "hydro.aux.rhine_dye": r"$\dot{Y}_e$",
            "hydro.aux.rhine_dyn": r"$\dot{Y}_n$",
            "hydro.aux.rhine_dyp": r"$\dot{Y}_p$",
            "hydro.aux.rhine_dya": r"$\dot{Y}_\alpha$",
            "hydro.aux.rhine_dyh": r"$\dot{Y}_h$",
            "hydro.aux.rhine_dah": r"$\dot{A}_h$",
            "hydro.aux.rhine_dma": r"$\dot{\tilde{m}}$",
            "hydro.aux.rhine_qphys": r"$\alpha\sqrt{\gamma}\,\dot{q}_{\mathrm{phys}}$",
            "hydro.aux.rhine_qexit": r"$\alpha\sqrt{\gamma}\,\dot{q}_{\mathrm{exit}}$",
            "hydro.aux.rhine_qreent": r"$\alpha\sqrt{\gamma}\,\dot{q}_{\mathrm{reent}}$",
            "hydro.aux.nse_state": r"NSE state",
            # history (.hst) columns
            "dt": r"$\Delta t$",
            "N_MeshBlock": r"$N_{\mathrm{MB}}$",
            "mass": r"$M_{\mathrm{b}}$",
            "max_rho": r"$\rho_{\max}$",
            "max_T": r"$T_{\max}$",
            "min_alpha": r"$\alpha_{\min}$",
            "num_c2p_fail": r"$N_{\mathrm{c2p\,fail}}$",
            "H-norm2": r"$||H||_2$",
            "M-norm2": r"$||M||_2$",
            "E_int": r"$E_{\mathrm{int}}$",
            "E_kin": r"$E_{\mathrm{kin}}$",
            "m_ej_geod": r"$M_{\mathrm{ej}}^{\mathrm{geod}}$",
            "m_ej_bern": r"$M_{\mathrm{ej}}^{\mathrm{bern}}$",
            "rhine-qdot": r"$\dot{Q}_{\mathrm{RHINE}}$",
            "rhine-Lfnu": r"$L_{\nu}^{f_{\nu}}$",
            "rhine-qphys": r"$\dot{Q}_{\mathrm{phys}}$",
            "rhine-qexit": r"$\dot{Q}_{\mathrm{exit}}$",
            "rhine-qreent": r"$\dot{Q}_{\mathrm{reent}}$",
            "m-nonNSE": r"$M_{w<1}$",
            "7-scalar": r"$E_{\mathrm{mix}}$",
            "max_abs_Xcons_err": r"$\max|\Delta X_{\mathrm{cons}}|$",
        }

    def get_label(self, field_name: str) -> str:
        """
        Return the LaTeX-formatted label for a field.

        If the field name is not found in the label dictionary,
        returns the field name unchanged.

        Args:
            field_name: The variable name to look up.

        Returns:
            LaTeX-formatted label string, or the field_name if not found.
        """
        return self._labels.get(field_name, field_name)

    def add_label(self, field_name: str, label: str) -> None:
        """
        Add or update a field label.

        Args:
            field_name: The variable name to add a label for.
            label: The LaTeX-formatted label string.
        """
        self._labels[field_name] = label
