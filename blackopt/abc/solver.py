import os
import abc
from typing import ClassVar, DefaultDict, SupportsFloat
from collections import defaultdict
import datetime

from ilya_ezplot import Metric
import dill

from blackopt.abc import Problem, Solution
from blackopt.config import get_rootdir
from blackopt.exceptions import BlackoptException, EarlyStopException


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Solver(abc.ABC):
    """Solver is a class that encapsulates an optimization strategy."""

    checkpoints_folder = "checkpoints"

    name: str = None
    best_solution: Solution = None

    def __init__(self, problem: Problem, solution_cls: ClassVar[Solution]):
        problem.eval_count = 0
        self.problem = problem
        solution_cls.problem = problem
        self.solution_cls = solution_cls
        self.best_solution: Solution = self.solution_cls.random_solution()
        self.metrics: DefaultDict[str, Metric] = keydefaultdict(
            lambda k: Metric(name=str(self), y_label=k, x_label="evaluations")
        )

    def record(self):
        for k, v in self.best_solution.metrics().items():
            self.record_metric(f"best_{k}", v)

    def record_metric(self, name: str, val: SupportsFloat):
        self.metrics[name].add_record(self.problem.eval_count, val)

    def solve(self, steps):
        self.problem.eval_count = 0
        try:
            while self.problem.eval_count < steps:
                self.step()
        except EarlyStopException:
            print(f"{self} finished optimization with an EarlyStopException.")
        self.salut()

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError()

    def checkpoint(self):
        path = os.path.join(get_rootdir(), self.checkpoints_folder, str(self.problem))
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        with open(os.path.join(path, timestamp), "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def restore_latest(problem: Problem) -> "Solver":
        directory = os.path.join(get_rootdir(), Solver.checkpoints_folder, str(problem))
        try:
            checkpoints = os.listdir(directory)
        except FileNotFoundError:
            raise BlackoptException(
                f"The checkpoint directory {directory} doesn't exist. Were any checkpoints made?"
            )
        else:
            if len(checkpoints) == 0:
                raise BlackoptException(
                    f"No checkpoints found in directory {directory}"
                )
            else:
                cp = sorted(checkpoints)[-1]
                with open(os.path.join(directory, cp), "rb") as f:
                    restored = dill.load(f)
                    assert isinstance(restored, Solver)
                    return restored

    def __str__(self):
        return str(self.name)

    def salut(self):
        print(f"{self} is Done with {self.problem.eval_count} evaluations.")
