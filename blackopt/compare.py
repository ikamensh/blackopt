from __future__ import annotations
from typing import List, TYPE_CHECKING, ClassVar
from collections import defaultdict

import pathos

if TYPE_CHECKING:
    from blackopt.abc import Solver
    from ilya_ezplot import Metric


class SolverFactory:
    def __init__(self, target_cls: ClassVar[Solver], *args, **kwargs):
        self.target_cls = target_cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.target_cls(*self.args, **self.kwargs)


def one_trial(steps: int, solver_constructor: SolverFactory) -> Metric:
    s = solver_constructor()
    print(s)
    s.solve(steps)
    return s.best_score_metric


def n_runs(trials: int, steps: int, solver: SolverFactory) -> Metric:

    pool = pathos.pools.ProcessPool()
    metrics = pool.map(lambda x: one_trial(steps, solver), "x" * trials)
    return sum(metrics)


def compare_solvers(
    trials: int, steps: int, solvers: List[SolverFactory]
) -> List[Metric]:
    pool = pathos.pools.ProcessPool()

    metrics: List[Metric] = pool.map(
        lambda solver: one_trial(steps, solver), solvers * trials
    )
    print(len(metrics))
    names_to_metrics = defaultdict(list)
    for metric in metrics:
        names_to_metrics[metric.name].append(metric)

    return [sum(lst) for lst in names_to_metrics.values()]


