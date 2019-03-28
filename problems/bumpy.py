from __future__ import annotations
from typing import List
import random
from functools import lru_cache

from problems.problem import Problem, Solution

import math
from collections import namedtuple

sin_expr = namedtuple("sin_expr", "i j k")

def evaluate_sin(solution: List[float], expr: sin_expr) -> float:
    arg = solution[expr.i] * solution[expr.j] + expr.k
    return math.sin(arg)

class BumpyProblem(Problem):
    """
    fitness = Sum ( sin(x_i * x_j + k) )
    """

    def __init__(self, n_dim: int, expressions: List[sin_expr]):

        self.n_dim = n_dim
        self.expressions = expressions
        self.eval_count = 0

    @staticmethod
    def random_problem(n_dim: int, n_expr: int):


        exprs = [sin_expr(random.randint(0,n_dim-1), random.randint(0,n_dim-1), random.random())
                 for _ in range(n_expr)]

        return BumpyProblem(n_dim, exprs)



    @lru_cache(maxsize=int(2**16) )
    def evaluate(self, s: BumpySolution) -> int:
        self.eval_count += 1
        return sum(evaluate_sin(s.values, expr) for expr in self.expressions)

    def random_solution(self) -> Solution:
        values = [random.random() for i in range(self.n_dim)]
        return BumpySolution(self, values)

    def __str__(self):
        return f"{self.__class__.__name__} {self.n_dim} {len(self.expressions)}"


class BumpySolution(Solution):
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

        return BumpySolution(self.problem, new_values)
