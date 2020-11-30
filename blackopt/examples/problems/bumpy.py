from typing import List
import random
from functools import lru_cache

from blackopt.abc import Problem, Solution

import math
from collections import namedtuple

sin_expr = namedtuple("sin_expr", "i j k")


def evaluate_sin(solution: List[float], expr: sin_expr) -> float:
    arg = solution[expr.i] * solution[expr.j] + expr.k
    return math.sin(arg)


class BumpyProblem(Problem):
    """
    The fitness landscape is a smooth surface with local optima.

    fitness = Sum ( sin(x_i * x_j + k) )
    """

    def __init__(self, n_dim: int, expressions: List[sin_expr]):

        self.n_dim = n_dim
        self.expressions = expressions
        self.score_span = 2 * len(expressions)
        self.eval_count = 0

    @staticmethod
    def random_problem(n_dim: int, n_expr: int):

        exprs = [
            sin_expr(
                random.randint(0, n_dim - 1),
                random.randint(0, n_dim - 1),
                random.random(),
            )
            for _ in range(n_expr)
        ]

        return BumpyProblem(n_dim, exprs)

    def evaluate(self, s: 'BumpySolution') -> int:
        self.eval_count += 1
        return sum(evaluate_sin(s.genes, expr) for expr in self.expressions)

    def __str__(self):
        return f"{self.__class__.__name__} {self.n_dim} {len(self.expressions)}"


class BumpySolution(Solution):
    problem: BumpyProblem

    def __init__(self, values):
        self.genes = values

    @staticmethod
    def random_solution() -> Solution:
        values = [random.random() for i in range(BumpySolution.problem.n_dim)]
        return BumpySolution(values)

    def mutate(self, rate: float):
        new_values = []
        for v in self.genes:
            if random.random() < rate:
                new_values.append(random.random())
            else:
                new_values.append(v)

        return BumpySolution(new_values)

    def crossover(self, other: 'BumpySolution'):
        crossover_point = random.randint(1, len(self.genes) - 1)

        child_a = self.genes[:crossover_point] + other.genes[crossover_point:]
        child_b = other.genes[:crossover_point] + self.genes[crossover_point:]

        return [BumpySolution(child_a), BumpySolution(child_b)]

    @lru_cache(maxsize=512)
    def similarity(self, other: 'BumpySolution'):

        diff = 0
        for i in range(self.problem.n_dim):
            diff += abs(self.genes[i] - other.genes[i])

        return 1 - diff / self.problem.n_dim
