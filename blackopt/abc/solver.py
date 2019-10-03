from __future__ import annotations
import os
import abc
from typing import ClassVar, Dict, DefaultDict, SupportsFloat
from collections import defaultdict
import datetime

from ilya_ezplot import Metric
import dill

from blackopt.abc import Problem, Solution


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Solver(abc.ABC):
    checkpoints_folder = "_checkpoints"

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

        self.checkpoint_path = os.path.join(self.checkpoints_folder, str(self.problem))


    def record(self):
        solution_metric_dict = self.best_solution.metrics()
        for k, v in solution_metric_dict.items():
            self.record_metric(f"best_{k}", v)

    def record_metric(self, name: str, val: SupportsFloat):
        self.metrics[name].add_record(self.problem.eval_count, val)

    @abc.abstractmethod
    def solve(self, steps):
        raise NotImplementedError()

    def checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        with open(os.path.join(self.checkpoint_path, timestamp), 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def restore_latest(problem: Problem) -> Solver:
        directory = os.path.join(Solver.checkpoints_folder, str(problem))
        checkpoints = os.listdir(directory)

        if len(checkpoints) == 0:
            raise Exception(f"No checkpoints found in directory {directory}")
        else:
            cp = sorted(checkpoints)[-1]
            with open(os.path.join(directory, cp), 'rb') as f:
                restored = dill.load(f)
                assert isinstance(restored, Solver)
                return restored

    def __str__(self):
        return str(self.name)
