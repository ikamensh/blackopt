from blackopt.abc import Solution
from typing import List
import random
import copy

class EasySolution(Solution):

    @staticmethod
    def random_solution() -> Solution:
        return EasySolution()

    def mutate(self, rate: float) -> Solution:
        return copy.deepcopy(self)

    def crossover(self, other: Solution) -> List[Solution]:
        return [copy.deepcopy(random.choice([self, other]))]

def test_solution():
    sol = EasySolution()
    assert sol is not None