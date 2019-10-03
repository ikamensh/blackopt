from blackopt.abc import Solver
from typing import List, SupportsFloat
import random
import copy

from tests.test_problem import EasyProblem
from tests.test_solution import EasySolution


from blackopt.algorithms import RandomSearch

def test_random_search():
    rs = RandomSearch(EasyProblem(), EasySolution)
    rs.solve(100)

    assert isinstance(rs.best_solution.score, (int, float))


def test_checkpoint():
    rs = RandomSearch(EasyProblem(), EasySolution)
    best_score = rs.best_solution.score

    rs.checkpoint()

    restored = Solver.restore_latest(rs.problem)

    assert restored.best_solution.score == best_score