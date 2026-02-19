from typing import Type

from blackopt.abc import Problem, Solution, Solver


class HillClimber(Solver):
    name = "hill climb"

    def __init__(self, problem: Problem, solution_cls: Type[Solution], mutation_rate):

        assert 0 < mutation_rate <= 1
        self.mutation_rate = mutation_rate
        super().__init__(problem, solution_cls)

    def step(self):
        solution = self.best_solution.mutate(self.mutation_rate)
        if solution.score > self.best_solution.score:
            self.best_solution = solution
        self.record()

    def __str__(self):
        return f"{self.name} {self.mutation_rate}"
