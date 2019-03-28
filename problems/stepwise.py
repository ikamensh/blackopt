from __future__ import annotations
from typing import List
import random
from functools import lru_cache

from problems.problem import Problem, Solution


class StepProblem(Problem):

    def __init__(self, thresholds: List[float]):
        self.n_dim = len(thresholds)
        self.thresholds = thresholds
        for t in thresholds:
            assert 0 <= t <= 1
        self.eval_count = 0

    @staticmethod
    def random_problem(n_dim: int):
        thresholds = [random.random() for i in range(n_dim)]
        return StepProblem(thresholds)


    @staticmethod
    def _step_function( values: List[float], thresholds: List[float]) -> int:
        result = 0

        for val, threshold in zip(values, thresholds):
            if val > threshold: result += 1

        return result

    @lru_cache(maxsize=int(2**16) )
    def evaluate(self, s: StepSolution) -> int:
        self.eval_count += 1
        return self._step_function(s.values, self.thresholds)

    def random_solution(self) -> Solution:
        values = [random.random() for i in range(self.n_dim)]
        return StepSolution(self, values)

    def __str__(self):
        return f"{self.__class__.__name__} {self.n_dim}"


class StepSolution(Solution):
    def __init__(self, problem, values):
        self.problem = problem
        self.values = values

    @property
    def score(self):
        return self.problem.evaluate(self)

    def mutate(self, rate: float):
        new_values = []
        for v in self.values:
            if random.random() < rate:
                new_values.append(random.random())
            else:
                new_values.append(v)

        return StepSolution(self.problem, new_values)
