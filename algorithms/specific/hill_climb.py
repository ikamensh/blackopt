from problems.problem import Problem, Solution
from algorithms.solver import Solver


class HillClimber(Solver):
    name = "hill climb"

    def __init__(self, problem: Problem, mutation_rate):

        super().__init__(problem)
        self.mutation_rate = mutation_rate

        assert 0 < mutation_rate <= 1

    def solve(self, n_evaluations):

        doc_freq = 1 + n_evaluations // 500

        for i in range(n_evaluations):
            solution = self.best_solution.mutate(self.mutation_rate)
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if i % doc_freq == 0:
                self.record()


    def __str__(self):
        return f"{self.name} {self.mutation_rate}"
