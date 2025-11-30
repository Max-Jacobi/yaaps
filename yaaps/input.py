"""
Input module for parsing Athena++ input/parameter files.

This module provides utilities to read and interpret Athena++ parameter files,
supporting hierarchical section-based organization and various data types.
"""

import re
from collections.abc import Mapping
from typing import Any
import numpy as np

max_inp_len = 10000


class Input(Mapping):
    """
    A class for parsing and accessing Athena++ input/parameter files.

    This class reads Athena++ parameter files (typically .inp or .par files)
    and provides dictionary-like access to the parameters organized by sections.

    The file format consists of sections marked by <section_name> headers,
    followed by key = value pairs. Comments starting with # or // are ignored.

    Args:
        file_path: Path to the input/parameter file to read.

    Attributes:
        data: Dictionary containing parsed sections and their parameters.
        git_hash: Git hash extracted from the file header, if present.

    Raises:
        ValueError: If the file doesn't start with a section header or
            contains invalid lines.
        RuntimeError: If the file exceeds maximum length or can't be parsed.

    Example:
        >>> inp = Input("simulation.par")
        >>> inp["mesh/nx1"]  # Access nested parameter
        128
        >>> inp.keys()  # Get section names
        dict_keys(['job', 'mesh', 'meshblock', ...])
    """

    def __init__(self, file_path):
        self.data = {}
        current_section = None

        in_list = False
        with open(file_path, 'r', errors='replace') as inp:
            for _ in range(max_inp_len):
                line = inp.readline()
                if (line == "") or ("<par_end>" in line):
                    break

                if "GIT_HASH" in line:
                    self.git_hash = line.split("GIT_HASH: ")[-1].strip()
                    continue

                # Skip comments and empty lines
                if '#' in line:
                    line = line[:line.index('#')]
                if '//' in line:
                    line = line[:line.index('//')]
                line = line.strip()
                if not line:
                    continue

                if in_list:
                    value += " "+line
                else:
                    # Check if the line contains a section header
                    match_section = re.match(r'^<(\w+)>', line)
                    if match_section:
                        current_section = match_section.group(1)
                        self.data[current_section] = {}
                        continue

                    # Check if the line contains a parameter (key-value pair)
                    match_param = re.match(r'^(\w+)\s*=\s*(.+)$', line)
                    if current_section is None:
                        raise ValueError('File does not start with a section header')
                    if not match_param:
                        raise ValueError(f'Invalid line: "{line}"')

                    key = match_param.group(1)
                    value = match_param.group(2)

                in_list = value.startswith("[") and not value.endswith("]")

                if not in_list:
                    self.data[current_section][key] = self._interprete_value(value)
            else:
                raise RuntimeError("Could not read input file")

    def __getitem__(self, key):
        """
        Get a parameter value using slash-separated key notation.

        Args:
            key: Parameter key, optionally with section prefix separated by '/'.
                For example, 'mesh/nx1' accesses self.data['mesh']['nx1'].

        Returns:
            The parameter value.

        Raises:
            KeyError: If the key is not found.
        """
        keys = key.split('/')
        result = self.data
        for k in keys:
            result = result[k]
        return result

    def __iter__(self):
        """Return an iterator over the section names."""
        return iter(self.data)

    def __len__(self):
        """Return the number of sections in the input file."""
        return len(self.data)

    def keys(self):
        """Return the section names as dictionary keys."""
        return self.data.keys()

    @staticmethod
    def _interprete_value(value):
        """
        Parse a string value into the appropriate Python type.

        Attempts to convert the value to int, float, bool, or list.
        If none of these conversions work, returns the original string.

        Args:
            value: String value to interpret.

        Returns:
            Interpreted value as int, float, bool, list, or str.
        """
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return list(map(Input._interprete_value, value[1:-1].split(",")))
        for typ in (int, float):
            try:
                return typ(value)
            except ValueError:
                pass
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return value

    def diff(self, other: "Input", float_tol: float = 1e-8) -> dict[str, dict[str, Any]]:
        """
        Compare this input file with another and return the differences.

        Args:
            other: Another Input instance to compare against.
            float_tol: Relative tolerance for comparing floating-point values.
                Two floats are considered equal if 2*(v1-v2) < float_tol*(v1+v2).

        Returns:
            A nested dictionary where keys are section names and values are
            dictionaries of differing parameters. Each parameter entry contains
            a tuple (value_in_self, value_in_other). Missing values are
            represented as " - ".

        Example:
            >>> diff = inp1.diff(inp2)
            >>> diff['mesh']['nx1']
            (128, 256)  # inp1 has 128, inp2 has 256
        """
        diff = {}
        for grp in {*self.keys(), *other.keys()}:
            diff[grp] = {}
            grp1 = self.get(grp, {})
            grp2 = other.get(grp, {})
            for key in {*grp1.keys(), *grp2.keys()}:
                v1 = grp1.get(key, " - ")
                v2 = grp2.get(key, " - ")
                if not ((isinstance(v1 ,float)
                         and (isinstance(v2, float)
                         and 2*(v1-v2)<float_tol*(v1+v2)))
                        or v1==v2):
                        diff[grp][key] = (v1, v2)
            if len(diff[grp]) == 0:
                del diff[grp]
        return diff
