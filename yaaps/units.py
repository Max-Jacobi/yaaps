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
    "rho": (6.175828477586656e+17, " [g cm$^{-3}$]"),
    "eps": (8.9875517873681764e+20, " [erg g$^{-1}$]"),
    "P": (5.550725674743868e+38, " [erg cm$^{-3}$]"),
    # "mass": (1.988409870967742e+33, " [g]"),
    "energy": (1.7870936689836656e+53, " [erg]"),
    "time": (0.004925490948309319, " [ms]"),
    "r": (1.4766250382504018, " [km]"),
    "x": (1.4766250382504018, " [km]"),
    "y": (1.4766250382504018, " [km]"),
    "z": (1.4766250382504018, " [km]"),
    "mass": (1.0, r" [$M_\odot$]"),
    re.compile(r"nu\d_lum"): (3.628132869648639e+59, " [erg s$^{-1}$]"),
    re.compile(r"nu\d_en"): (1.11545707207968e+60, " [MeV]"),
    re.compile("m_ej"): (1.0, r" [$M_\odot$]"),
    re.compile("mdot_ej"): (203025.44670054692, r" [$M_\odot$ s$^{-1}$]"),
    re.compile("util_u"): (1.0, r" [$c$]"),
    re.compile("vel"): (1.0, r" [$c$]"),
    re.compile("x[1-3][vf]?"): (1.4766250382504018, " [km]"),
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
