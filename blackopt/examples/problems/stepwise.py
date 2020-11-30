from typing import List
import random
from functools import lru_cache

from blackopt.abc import Problem, Solution


class StepProblem(Problem):
    """The fitness is a sum of step functions for each dimension.

    The [1]*n_dim is aways an optimal solution, fitness is in range [0, n_dim].
    """

    def __init__(self, thresholds: List[float]):
        self.n_dim = len(thresholds)
        self.score_span = len(thresholds)
        self.thresholds = thresholds
        for t in thresholds:
            assert 0 <= t <= 1
        self.eval_count = 0

    @staticmethod
    def random_problem(n_dim: int):
        thresholds = [random.random() for i in range(n_dim)]
        return StepProblem(thresholds)

    @staticmethod
    def _step_function(values: List[float], thresholds: List[float]) -> int:
        result = 0

        for val, threshold in zip(values, thresholds):
            if val > threshold:
                result += 1

        return result

    def evaluate(self, s: "StepSolution") -> int:
        self.eval_count += 1
        return self._step_function(s.genes, self.thresholds)

    def __str__(self):
        return f"{self.__class__.__name__} {self.n_dim}"


class StepSolution(Solution):
    problem: StepProblem

    def __init__(self, values):
        self.genes = values

    @staticmethod
    def random_solution() -> Solution:
        values = [random.random() for i in range(StepSolution.problem.n_dim)]
        return StepSolution(values)

    def mutate(self, rate: float):
        new_values = []
        for v in self.genes:
            if random.random() < rate:
                new_values.append(random.random())
            else:
                new_values.append(v)

        return StepSolution(new_values)

    def crossover(self, other: "StepSolution"):
        crossover_point = random.randint(1, len(self.genes) - 1)

        child_a = self.genes[:crossover_point] + other.genes[crossover_point:]
        child_b = other.genes[:crossover_point] + self.genes[crossover_point:]

        return [StepSolution(child_a), StepSolution(child_b)]

    @lru_cache(maxsize=512)
    def similarity(self, other: "StepSolution"):

        diff = 0
        for i in range(self.problem.n_dim):
            diff += abs(self.genes[i] - other.genes[i])

        return 1 - diff / self.problem.n_dim
