from blackopt.algorithms import Rapga
from blackopt.abc import Problem, Solver
import random

import pathos


class Sasegasa(Solver):
    def __init__(
            self,
            problem: Problem,
            solution_cls,
            popsize: int,
            mutation_rate: float,
            elite_size: int,
            equal_chances: float = 0.5,
            growth_factor=30
    ):
        self.problem = problem,
        self.solution_cls = solution_cls
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.equal_chances = equal_chances
        self.growth_factor = growth_factor

        super().__init__(problem, solution_cls)

        self.pool = pathos.pools.ProcessPool()


