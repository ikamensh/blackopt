from __future__ import annotations
import abc
import os
import dill
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from blackopt.abc import Solution

import uuid


class Problem(abc.ABC):
    eval_count: int = None
    score_span: float = 1

    store_dir = "_problems"

    @abc.abstractmethod
    def evaluate(self, s: Solution) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()

    def save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        identifier = str(self)

        existing = os.listdir(self.store_dir)
        if identifier in existing:
            identifier += uuid.uuid4()

        with open(os.path.join(self.store_dir, identifier), 'wb') as f:
            dill.dump(self, f)
            print(f"Stored problem as {identifier}")

    @staticmethod
    def load(identifier: str) -> Problem:
        directory = os.path.join(Problem.store_dir)

        with open(os.path.join(directory, identifier), 'rb') as f:
            restored = dill.load(f)
            assert isinstance(restored, Problem)
            return restored


