"""
Plot formatting module for GRAthena++ simulation visualizations.

This module provides abstract base class PlotFormatter and concrete subclasses
RawPlotFormatter and PaperPlotFormatter for handling plot labels and unit
conversions.

- RawPlotFormatter: Uses code-internal naming and no unit conversions
- PaperPlotFormatter: Uses LaTeX labels with unit conversions for publication-ready figures

Example:
    >>> from yaaps.plot_formatter import PlotFormatter, RawPlotFormatter, PaperPlotFormatter
    >>> formatter = PlotFormatter(mode="paper")  # Factory creates PaperPlotFormatter
    >>> formatter.format_axis_label("rho")
    '$\\rho$ [g cm$^{-3}$]'
    >>> raw_formatter = RawPlotFormatter()
    >>> raw_formatter.format_axis_label("rho")
    'rho'
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from .units import UnitConverter, FieldLabels


PlotMode = Literal["raw", "paper"]
"""Type alias for plot formatting modes.

Available modes:
    - "raw": Use code-internal variable names and code units
    - "paper": Use LaTeX labels and physical units for publication-ready figures
"""


class PlotFormatterBase(ABC):
    """
    Abstract base class for plot label formatting and unit conversions.

    This class defines the interface for all formatters. Subclasses must
    implement the abstract methods to provide mode-specific formatting behavior.

    Attributes:
        mode: The formatting mode identifier ("raw" or "paper").
        unit_converter: UnitConverter instance for unit conversions.
        field_labels: FieldLabels instance for LaTeX label mappings.
    """

    mode: PlotMode
    unit_converter: UnitConverter
    field_labels: FieldLabels

    # Format strings for titles
    _title_format: str
    _time_title_format: str

    def __init__(
        self,
        unit_converter: UnitConverter | None = None,
        field_labels: FieldLabels | None = None,
    ):
        """
        Initialize the PlotFormatter.

        Args:
            unit_converter: Optional UnitConverter instance. If None, uses
                the default UnitConverter.
            field_labels: Optional FieldLabels instance. If None, uses
                the default FieldLabels.
        """
        self.unit_converter = unit_converter if unit_converter is not None else UnitConverter()
        self.field_labels = field_labels if field_labels is not None else FieldLabels()

    @abstractmethod
    def format_axis_label(self, field_name: str) -> str:
        """
        Format an axis label for the given field.

        Args:
            field_name: The variable name to format (e.g., "rho", "x1v").

        Returns:
            Formatted axis label string.
        """
        ...

    @abstractmethod
    def format_title(self, field_name: str, time: float) -> str:
        """
        Format a plot title with field name and time.

        Args:
            field_name: The variable name to format.
            time: Simulation time in code units.

        Returns:
            Formatted title string.
        """
        ...

    @abstractmethod
    def format_colorbar_label(self, field_name: str) -> str:
        """
        Format a colorbar label for the given field.

        Args:
            field_name: The variable name to format.

        Returns:
            Formatted colorbar label string.
        """
        ...

    @abstractmethod
    def convert_data(self, field_name: str, data: np.ndarray) -> np.ndarray:
        """
        Convert data values to display units.

        Args:
            field_name: The variable name for unit lookup.
            data: NumPy array of data values in code units.

        Returns:
            NumPy array of data values in display units.
        """
        ...

    @abstractmethod
    def convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """
        Convert coordinate values to display units.

        Args:
            coord_name: The coordinate name (e.g., "x1v", "x2v", "x3v").
            coord_data: NumPy array of coordinate values in code units.

        Returns:
            NumPy array of coordinate values in display units.
        """
        ...

    @abstractmethod
    def inverse_convert_time(self, time: float) -> float:
        """
        Convert time from display units back to code units.

        Args:
            time: Time value in display units.

        Returns:
            Time value in code units.
        """
        ...

    @abstractmethod
    def inverse_convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """
        Convert coordinate values from display units back to code units.

        Args:
            coord_name: The coordinate name (e.g., "x1v", "x2v", "x3v").
            coord_data: NumPy array of coordinate values in display units.

        Returns:
            NumPy array of coordinate values in code units.
        """
        ...


class RawPlotFormatter(PlotFormatterBase):
    """
    Formatter for raw mode with code-internal naming and no unit conversions.

    This formatter returns field names as-is and performs no unit conversions
    on data or coordinates.

    Example:
        >>> formatter = RawPlotFormatter()
        >>> formatter.format_axis_label("x1v")
        'x1v'
        >>> formatter.format_title("rho", time=100.0)
        'rho @ t= 100.00'
    """

    mode: PlotMode = "raw"
    _title_format: str = "{field_name} @ t= {time:.2f}"

    def format_axis_label(self, field_name: str) -> str:
        """Return the field name as-is."""
        return field_name

    def format_title(self, field_name: str, time: float) -> str:
        """Format title with raw field name and time."""
        return self._title_format.format(field_name=field_name, time=time)

    def format_colorbar_label(self, field_name: str) -> str:
        """Return the field name as-is."""
        return field_name

    def convert_data(self, field_name: str, data: np.ndarray) -> np.ndarray:
        """Return data unchanged (identity transformation)."""
        return data

    def convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """Return coordinates unchanged (identity transformation)."""
        return coord_data

    def inverse_convert_time(self, time: float) -> float:
        """Return time unchanged (identity transformation)."""
        return time

    def inverse_convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """Return coordinates unchanged (identity transformation)."""
        return coord_data


class PaperPlotFormatter(PlotFormatterBase):
    """
    Formatter for paper mode with LaTeX labels and physical unit conversions.

    This formatter returns LaTeX-formatted labels with units and converts
    data and coordinates from code units to physical units.

    Example:
        >>> formatter = PaperPlotFormatter()
        >>> formatter.format_axis_label("x1v")
        '$x$ [km]'
        >>> formatter.format_title("rho", time=100.0)
        '$\\rho$ @ $t$ = 0.49 ms'
    """

    mode: PlotMode = "paper"
    _label_unit_format: str = "{label}{unit}"
    _title_format: str = "{label} @ {time_label} = {time:.2f} {time_unit}"

    def format_axis_label(self, field_name: str) -> str:
        """Return LaTeX-formatted label with units."""
        label = self.field_labels.get_label(field_name)
        _, unit = self.unit_converter.get_conversion(field_name)
        if unit:
            return self._label_unit_format.format(label=label, unit=unit)
        return label

    def format_title(self, field_name: str, time: float) -> str:
        """Format title with LaTeX label and converted time."""
        label = self.field_labels.get_label(field_name)
        time_scale, time_unit = self.unit_converter.get_conversion("time")
        converted_time = time * time_scale
        time_label = self.field_labels.get_label("time")
        time_unit_clean = time_unit.strip() if time_unit else ""
        return self._title_format.format(
            label=label,
            time_label=time_label,
            time=converted_time,
            time_unit=time_unit_clean,
        )

    def format_colorbar_label(self, field_name: str) -> str:
        """Return LaTeX-formatted label with units."""
        label = self.field_labels.get_label(field_name)
        _, unit = self.unit_converter.get_conversion(field_name)
        if unit:
            return self._label_unit_format.format(label=label, unit=unit)
        return label

    def convert_data(self, field_name: str, data: np.ndarray) -> np.ndarray:
        """Scale data by the appropriate conversion factor."""
        scale, _ = self.unit_converter.get_conversion(field_name)
        return data * scale

    def convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """Scale coordinates by the appropriate conversion factor."""
        scale, _ = self.unit_converter.get_conversion(coord_name)
        return coord_data * scale

    def inverse_convert_time(self, time: float) -> float:
        """Convert time from physical units back to code units."""
        scale, _ = self.unit_converter.get_conversion("time")
        return time / scale

    def inverse_convert_coordinate(self, coord_name: str, coord_data: np.ndarray) -> np.ndarray:
        """Convert coordinates from physical units back to code units."""
        scale, _ = self.unit_converter.get_conversion(coord_name)
        return coord_data / scale


def PlotFormatter(
    mode: PlotMode = "raw",
    unit_converter: UnitConverter | None = None,
    field_labels: FieldLabels | None = None,
) -> PlotFormatterBase:
    """
    Factory function to create an appropriate PlotFormatter subclass.

    This function provides backward compatibility with the previous class-based API.

    Args:
        mode: Formatting mode, either "raw" or "paper". Defaults to "raw".
        unit_converter: Optional UnitConverter instance. If None, uses
            the default UnitConverter.
        field_labels: Optional FieldLabels instance. If None, uses
            the default FieldLabels.

    Returns:
        RawPlotFormatter if mode is "raw", PaperPlotFormatter if mode is "paper".

    Example:
        >>> formatter = PlotFormatter(mode="paper")
        >>> isinstance(formatter, PaperPlotFormatter)
        True
        >>> formatter.format_axis_label("rho")
        '$\\rho$ [g cm$^{-3}$]'
    """
    if mode == "raw":
        return RawPlotFormatter(unit_converter=unit_converter, field_labels=field_labels)
    else:
        return PaperPlotFormatter(unit_converter=unit_converter, field_labels=field_labels)
