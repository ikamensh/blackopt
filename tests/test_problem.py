from blackbox.abc import Solution, Problem
import random

class EasyProblem(Problem):

    def evaluate(self, s: Solution) -> float:
        return random.random()

    def __str__(self):
        return "Very easy problem"

def test_problem():
    ep = EasyProblem()
