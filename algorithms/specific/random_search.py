from problems.problem import Problem, Solution
from algorithms.solver import Solver
from utils.plot import Metric


class RandomSearch(Solver):
    name = "random search"

    def __init__(self, problem: Problem):
        self.problem = problem
        self.best_solution: Solution = problem.random_solution()

    def solve(self, n_evaluations):

        best_score_metric = Metric(x_label="evaluations", y_label="best_score")
        doc_freq = max(1, n_evaluations // 1000)

        while self.problem.eval_count < n_evaluations:
            solution = self.problem.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if self.problem.eval_count % doc_freq == 0:
                best_score_metric.add_record(
                    self.problem.eval_count, self.best_solution.score)

        return best_score_metric
