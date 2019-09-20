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


    def record_result(self):
        self.best_score_metric.add_record(self.problem.eval_count, self.best_solution.score)

    @abc.abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.best_solution: Solution = self.problem.random_solution()
        self.problem.eval_count = 0

    def __str__(self):
        return str(self.name)
