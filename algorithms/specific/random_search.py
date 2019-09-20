from problems.problem import Problem, Solution
from algorithms.solver import Solver


class RandomSearch(Solver):
    name = "random search"

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.best_solution: Solution = problem.random_solution()

    def solve(self, n_evaluations):

        doc_freq = 1 + n_evaluations // 500

        for i in range(n_evaluations):
            solution = self.problem.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if not i % doc_freq:
                print(i)
                self.record()

