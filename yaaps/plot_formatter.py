"""
Plot formatting module for GRAthena++ simulation visualizations.

This module provides the PlotFormatter class for handling plot labels and unit
conversions in two modes: "raw" mode (code-internal naming) and "paper" mode
(LaTeX labels with unit conversions for publication-ready figures).

Example:
    >>> from yaaps.plot_formatter import PlotFormatter
    >>> formatter = PlotFormatter(mode="paper")
    >>> formatter.format_axis_label("rho")
    '$\\rho$ [g cm$^{-3}$]'
    >>> formatter.set_mode("raw")
    >>> formatter.format_axis_label("rho")
    'rho'
"""

from typing import Literal

import numpy as np

from .units import UnitConverter, FieldLabels


PlotMode = Literal["raw", "paper"]
"""Type alias for plot formatting modes.

Available modes:
    - "raw": Use code-internal variable names and code units
    - "paper": Use LaTeX labels and physical units for publication-ready figures
"""


class PlotFormatter:
    """
    Formatter class for plot labels and unit conversions.

    This class handles all label formatting and data conversions for plots,
    supporting both "raw" mode (code-internal naming) and "paper" mode
    (LaTeX labels with physical units).

    Attributes:
        mode: Current formatting mode, either "raw" or "paper".
        unit_converter: UnitConverter instance for unit conversions.
        field_labels: FieldLabels instance for LaTeX label mappings.

    Example:
        >>> formatter = PlotFormatter(mode="paper")
        >>> formatter.format_axis_label("x1v", axis="x")
        '$x$ [km]'
        >>> formatter.format_title("rho", time=100.0)
        '$\\rho$ @ $t$ = 0.49 ms'
    """

    def __init__(
        self,
        mode: PlotMode = "raw",
        unit_converter: UnitConverter | None = None,
        field_labels: FieldLabels | None = None,
    ):
        """
        Initialize the PlotFormatter.

        Args:
            mode: Formatting mode, either "raw" or "paper". Defaults to "raw".
            unit_converter: Optional UnitConverter instance. If None, uses
                the default UnitConverter.
            field_labels: Optional FieldLabels instance. If None, uses
                the default FieldLabels.

        Example:
            >>> formatter = PlotFormatter(mode="paper")
            >>> formatter.mode
            'paper'
        """
        self.mode: PlotMode = mode
        self.unit_converter = unit_converter if unit_converter is not None else UnitConverter()
        self.field_labels = field_labels if field_labels is not None else FieldLabels()

    def set_mode(self, mode: PlotMode) -> None:
        """
        Change the formatting mode.

        Args:
            mode: New formatting mode, either "raw" or "paper".

        Example:
            >>> formatter = PlotFormatter(mode="raw")
            >>> formatter.set_mode("paper")
            >>> formatter.mode
            'paper'
        """
        self.mode = mode

    def format_axis_label(self, field_name: str) -> str:
        """
        Format an axis label based on the current mode.

        In "raw" mode, returns the field name as-is (or axis if provided).
        In "paper" mode, returns a LaTeX-formatted label with units.

        Args:
            field_name: The variable name to format (e.g., "rho", "x1v").

        Returns:
            Formatted axis label string.

        Example:
            >>> formatter = PlotFormatter(mode="paper")
            >>> formatter.format_axis_label("x1v")
            '$x$ [km]'
            >>> formatter.set_mode("raw")
            >>> formatter.format_axis_label("x1v")
            'x1v'
        """
        if self.mode == "raw":
            return field_name
        else:
            return self._format_axis_label_paper(field_name)

    def format_title(self, field_name: str, time: float) -> str:
        """
        Format a plot title with proper time units.

        In "raw" mode, returns "field_name @ t= time".
        In "paper" mode, returns "LaTeX_label @ $t$ = converted_time unit".

        Args:
            field_name: The variable name to format.
            time: Simulation time in code units.

        Returns:
            Formatted title string.

        Example:
            >>> formatter = PlotFormatter(mode="paper")
            >>> formatter.format_title("rho", time=100.0)
            '$\\rho$ @ $t$ = 0.49 ms'
            >>> formatter.set_mode("raw")
            >>> formatter.format_title("rho", time=100.0)
            'rho @ t= 100.00'
        """
        if self.mode == "raw":
            return f"{field_name} @ t= {time:.2f}"
        else:
            return self._format_title_paper(field_name, time)

    def format_colorbar_label(self, field_name: str) -> str:
        """
        Format a colorbar label based on the current mode.

        In "raw" mode, returns the field name.
        In "paper" mode, returns a LaTeX-formatted label with units.

        Args:
            field_name: The variable name to format.

        Returns:
            Formatted colorbar label string.

        Example:
            >>> formatter = PlotFormatter(mode="paper")
            >>> formatter.format_colorbar_label("rho")
            '$\\rho$ [g cm$^{-3}$]'
            >>> formatter.set_mode("raw")
            >>> formatter.format_colorbar_label("rho")
            'rho'
        """
        if self.mode == "raw":
            return field_name
        else:
            return self._format_colorbar_label_paper(field_name)

    def convert_data(self, field_name: str, data: np.ndarray) -> np.ndarray:
        """
        Convert data values based on the current mode.

        In "raw" mode, returns data unchanged (identity transformation).
        In "paper" mode, scales data by the appropriate conversion factor.

        Args:
            field_name: The variable name for unit lookup.
            data: NumPy array of data values in code units.

        Returns:
            NumPy array of data values (in code units for raw mode,
            in physical units for paper mode).

        Example:
            >>> import numpy as np
            >>> formatter = PlotFormatter(mode="paper")
            >>> data = np.array([1.0, 2.0, 3.0])
            >>> converted = formatter.convert_data("rho", data)
            >>> # Data scaled by density conversion factor
        """
        if self.mode == "raw":
            return data
        else:
            scale, _ = self.unit_converter.get_conversion(field_name)
            return data * scale

    def convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """
        Convert coordinate values based on the current mode.

        In "raw" mode, returns coordinates unchanged.
        In "paper" mode, scales coordinates by the appropriate factor (e.g., to km).

        Args:
            coord_name: The coordinate name (e.g., "x1v", "x2v", "x3v").
            coord_data: NumPy array of coordinate values in code units.

        Returns:
            NumPy array of coordinate values (in code units for raw mode,
            in physical units for paper mode).

        Example:
            >>> import numpy as np
            >>> formatter = PlotFormatter(mode="paper")
            >>> coords = np.array([0.0, 10.0, 20.0])
            >>> converted = formatter.convert_coordinate("x1v", coords)
            >>> # Coordinates scaled to km
        """
        if self.mode == "raw":
            return coord_data
        else:
            scale, _ = self.unit_converter.get_conversion(coord_name)
            return coord_data * scale

    def _format_axis_label_paper(self, field_name: str) -> str:
        """
        Format axis label in paper mode with LaTeX and units.

        Args:
            field_name: The variable name to format.

        Returns:
            LaTeX-formatted label with units.
        """
        label = self.field_labels.get_label(field_name)
        _, unit = self.unit_converter.get_conversion(field_name)
        if unit:
            return f"{label}{unit}"
        return label

    def _format_title_paper(self, field_name: str, time: float) -> str:
        """
        Format title in paper mode with LaTeX and converted time.

        Args:
            field_name: The variable name.
            time: Simulation time in code units.

        Returns:
            LaTeX-formatted title with converted time and units.
        """
        label = self.field_labels.get_label(field_name)
        time_scale, time_unit = self.unit_converter.get_conversion("time")
        converted_time = time * time_scale
        time_label = self.field_labels.get_label("time")
        # Remove leading space from unit if present
        time_unit_clean = time_unit.strip() if time_unit else ""
        return f"{label} @ {time_label} = {converted_time:.2f} {time_unit_clean}"

    def _format_colorbar_label_paper(self, field_name: str) -> str:
        """
        Format colorbar label in paper mode with LaTeX and units.

        Args:
            field_name: The variable name to format.

        Returns:
            LaTeX-formatted label with units.
        """
        label = self.field_labels.get_label(field_name)
        _, unit = self.unit_converter.get_conversion(field_name)
        if unit:
            return f"{label}{unit}"
        return label
