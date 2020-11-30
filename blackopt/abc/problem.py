import abc
import os
import dill
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blackopt.abc import Solution

import uuid
from blackopt.config import get_rootdir


class Problem(abc.ABC):
    eval_count: int = None
    score_span: float = 1
    store_dir = "problems"
    name: str

    @abc.abstractmethod
    def evaluate(self, s: "Solution") -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()

    def save(self) -> str:
        folder = os.path.join(get_rootdir(), self.store_dir)
        os.makedirs(folder, exist_ok=True)
        identifier = str(self)

        existing = os.listdir(folder)
        if identifier in existing:
            identifier += str(uuid.uuid4())

        with open(os.path.join(folder, identifier), "wb") as f:
            dill.dump(self, f)
            print(f"Stored problem as '{identifier}' at {folder}")

        return identifier

    @staticmethod
    def load(identifier: str) -> "Problem":
        directory = os.path.join(get_rootdir(), Problem.store_dir)

        with open(os.path.join(directory, identifier), "rb") as f:
            restored = dill.load(f)
            assert isinstance(restored, Problem)
            return restored
