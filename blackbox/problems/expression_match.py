from __future__ import annotations
import random
import time

from problems.problem import Problem, Solution
from typing import Callable

import numpy as np

from functools import lru_cache


from expr_tree import ExpressionTree

from blackbox.util.timeout import timeout, UserTimeoutError

import math


class SinGpSolution(Solution):
    def __init__(self, problem, expr_tree: ExpressionTree):
        self.problem = problem
        self.tree = expr_tree
        self.time = None

    @property
    @timeout(0.1)
    def score(self):
        try:
            return self.problem.evaluate(self)
        except UserTimeoutError:
            return -math.inf

    @timeout(0.1)
    def mutate(self, rate: float):
        try:
            return SinGpSolution(self.problem, self.tree.mutate(rate))
        except UserTimeoutError:
            self.tree.penalty += 100
            return self

    @timeout(0.1)
    def crossover(self, other: SinGpSolution):
        try:
            return [
                SinGpSolution(self.problem, ExpressionTree.crossover(self.tree, other.tree))
            ]
        except UserTimeoutError:
            self.tree.penalty += 100
            return [self]


    def metrics(self):
        return {
            "nodes": len(self.tree.nodes),
            "penalty": self.tree.penalty,
            "time": self.time or 0,
        }


class ExpressionMatchProblem(Problem):

    solution_cls = SinGpSolution

    def __init__(self, expression: Callable[[float], float], xmin, xmax):
        """
        The problem of constructing an approximation to f(x) = a * sin(w * x + k)
        subject to xmin < x < xmax
        """

        self.X = np.arange(xmin, xmax, (xmax - xmin) / 100)
        self.Y = np.array([expression(x) for x in self.X])
        self.resolution = len(self.X)
        self.eval_count = 0

    @staticmethod
    def random_problem(*args):
        raise NotImplementedError()

    @lru_cache(maxsize=int(2 ** 10))
    def evaluate(self, s: SinGpSolution) -> float:
        self.eval_count += 1

        # idx = np.random.randint(0, self.resolution, size=int( (self.resolution) ** (1/2) ) )
        # x_test, y_test = self.X[idx], self.Y[idx]

        t = time.clock()
        y_pred = [s.tree.eval(x) for x in self.X]
        s.time = time.clock() - t

        y_pred = np.array(y_pred)
        errors = self.Y - y_pred
        errors = errors ** 2

        return -np.mean(np.log(errors)) - 0.001 * ( abs(len(s.tree.nodes) - 10 ) + s.tree.penalty)

    def random_solution(self) -> Solution:
        depth = random.randint(2, 8)
        return SinGpSolution(self, ExpressionTree.fill(depth))

    def __str__(self):
        return f"{self.__class__.__name__}"
