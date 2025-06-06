import re
from collections.abc import Mapping
from typing import Any
import numpy as np

max_inp_len = 1000

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

        Returns a dictionary with three keys:
        - 'added': keys present in d2 but not in d1, with their values from d2.
        - 'removed': keys present in d1 but not in d2, with their values from d1.
        - 'changed': keys present in both but with different values, mapping to a tuple (d1_value, d2_value).
        """
        d1 = {
            f'{gk}/{sk}': self[gk][sk]
            for gk in sorted(self.keys())
            for sk in sorted(self[gk])
            }
        d2 = {
            f'{gk}/{sk}': other[gk][sk]
            for gk in sorted(other.keys())
            for sk in sorted(other[gk])
            }

        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        added = {k: d2[k] for k in keys2 - keys1}
        removed = {k: d1[k] for k in keys1 - keys2}
        common = keys1 & keys2

        def _comp(v1, v2):
            if isinstance(v1, float):
                return np.isclose(v1, v2, rtol=float_tol)
            if isinstance(v1, list):
                if len(v1) != len(v2):
                    return False
                return all(_comp(vv1, vv2) for vv1, vv2 in zip(v1, v2))
            if isinstance(v1, str):
                return v1.strip() == v2.strip()
            return v1 == v2

        changed = {k: (d1[k], d2[k]) for k in common if not _comp(d1[k], d2[k])}

        return {
            'added': added,
            'removed': removed,
            'changed': changed
        }
