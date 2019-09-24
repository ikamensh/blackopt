import abc
from typing import Dict, ClassVar
from blackopt.abc import Problem, Solution
from ilya_ezplot import Metric


class Solver(abc.ABC):
    name: str = None
    best_solution: Solution = None

    def __init__(
        self,
        problem: Problem,
        solution_cls: ClassVar[Solution],
        plot_kwargs: Dict = None,
    ):
        problem.eval_count = 0
        self.problem = problem
        self.best_score_metric = Metric(
            name=str(self),
            x_label="evaluations",
            y_label="best_score",
            style_kwargs=plot_kwargs or {},
        )

        solution_cls.problem = problem
        self.solution_cls = solution_cls
        self.best_solution: Solution = self.solution_cls.random_solution()

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
    def solve(self, steps):
        raise NotImplementedError()

    def __str__(self):
        return str(self.name)
