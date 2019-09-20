from problems.problem import Problem, Solution
from algorithms.solver import Solver
from ilya_ezplot import Metric


class RandomSearch(Solver):
    name = "random search"

    def __init__(self, problem: Problem):
        self.problem = problem
        self.best_solution: Solution = problem.random_solution()

    def solve(self, n_evaluations):

        best_score_metric = Metric(name=self.name, x_label="evaluations", y_label="best_score")
        doc_freq = 1 + n_evaluations // 500

        for i in range(n_evaluations):
            solution = self.problem.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if not i % doc_freq:
                print(i)
                best_score_metric.add_record(
                    self.problem.eval_count, self.best_solution.score)

        return best_score_metric
