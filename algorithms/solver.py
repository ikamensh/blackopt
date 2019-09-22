import abc
from typing import Dict
from problems.problem import Problem, Solution
from ilya_ezplot import Metric


class Solver(abc.ABC):
    name: str = None
    best_solution: Solution = None

    def __init__(self, problem: Problem, plot_kwargs: Dict = None):
        self.problem = problem
        self.best_score_metric = Metric(
            name=self.name,
            x_label="evaluations",
            y_label="best_score",
            style_kwargs=plot_kwargs or {},
        )
        self.best_solution: Solution = problem.random_solution()

        solution_metric_dict = self.best_solution.metrics()
        self.solution_metrics = {
            k: Metric(name=k, x_label="evaluations")
            for k in solution_metric_dict.keys()
        }

        self.record()

    def record(self):
        self.best_score_metric.add_record(
            self.problem.eval_count, self.best_solution.score
        )

        solution_metric_dict = self.best_solution.metrics()

        for k, v in solution_metric_dict.items():
            self.solution_metrics[k].add_record(self.problem.eval_count, v)

    @abc.abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.best_solution: Solution = self.problem.random_solution()
        self.problem.eval_count = 0

    def __str__(self):
        return str(self.name)
