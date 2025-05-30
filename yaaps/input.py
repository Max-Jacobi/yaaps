import re
from collections.abc import Mapping


class Input(Mapping):
    def __init__(self, file_path):
        self.data = {}
        current_section = None

        with open(file_path, 'r') as file:
            lines = file.readlines()

        in_list = False
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if '#' in line:
                line = line[:line.index('#')]
            if '//' in line:
                line = line[:line.index('//')]
            if not line:
                continue
            line = line.strip()

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
                    raise ValueError(f'Invalid line: {line}')

                key = match_param.group(1)
                value = match_param.group(2)

            in_list = value.startswith("[") and not value.endswith("]")

            if not in_list:
                self.data[current_section][key] = self._interprete_value(value)

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
