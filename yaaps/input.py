import re
from collections.abc import Mapping
from typing import Any
import numpy as np

max_inp_len = 10000

class Input(Mapping):
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
        keys = key.split('/')
        result = self.data
        for k in keys:
            result = result[k]
        return result

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    @staticmethod
    def _interprete_value(value):
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
        Compare two dictionaries and return the differences.
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
