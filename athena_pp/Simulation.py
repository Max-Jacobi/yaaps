import os
from typing import Optional
from functools import lru_cache

from .Input import Input
from .athena_read import hst, athdf


class Simulation:
    path: str
    problem_id: str
    input: Input
    name: str

    def __init__(
        self,
        path: str,
        input_path: Optional[str] = None
    ):
        self.path = path
        self.name = os.path.basename(path)
        if input_path is None:
            for file in os.listdir(path):
                if file.endswith('.inp'):
                    input_path = os.path.join(path, file)
                    break
                elif file.endswith('.par'):
                    input_path = os.path.join(path, file)
                    break
        self.input = Input(input_path)
        self.problem_id = self.input['job/problem_id']

    @property
    @lru_cache(maxsize=None)
    def hst(self, **kwargs) -> dict:
        return hst(f"{self.path}/{self.problem_id}.hst", **kwargs)

    @lru_cache(maxsize=128)
    def get_athdf(self, output: int, it: int, **kwargs) -> dict:
        return athdf(f"{self.path}/{self.problem_id}.out{output}.{it:05d}.athdf", **kwargs)
