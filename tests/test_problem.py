from blackopt.abc import Solution, Problem
import random


class EasyProblem(Problem):
    def __init__(self):
        self.secret_number = random.randint(0, 1000)

    def evaluate(self, s: Solution) -> float:
        return random.random()

    def __str__(self):
        return "Very easy problem"


def test_problem():
    ep = EasyProblem()


def test_storing():
    """Verify a problem can be saved into a file. """

    ep = EasyProblem()
    identifier = ep.save()

    loaded: EasyProblem = EasyProblem.load(identifier)

    assert isinstance(loaded, EasyProblem)
    assert ep.secret_number == loaded.secret_number
