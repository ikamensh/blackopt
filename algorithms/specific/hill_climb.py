from problems.problem import Problem, Solution
from algorithms.solver import Solver
from ilya_ezplot import Metric


class HillClimber(Solver):
    name = "hill climb"

    def __init__(self, problem: Problem, mutation_rate):

        self.mutation_rate = mutation_rate
        self.problem = problem
        self.best_solution: Solution = problem.random_solution()

        assert 0 < mutation_rate <= 1

    def solve(self, n_evaluations):

        best_score_metric = Metric(x_label="evaluations", y_label="best_score")
        doc_freq = max(1, n_evaluations // 500)

        while self.problem.eval_count < n_evaluations:
            solution = self.best_solution.mutate(self.mutation_rate)
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if self.problem.eval_count % doc_freq == 0:
                best_score_metric.add_record(
                    self.problem.eval_count, self.best_solution.score)

        return best_score_metric

    def __str__(self):
        return f"{self.name} {self.mutation_rate}"
