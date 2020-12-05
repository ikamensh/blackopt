from typing import List, TYPE_CHECKING, Dict, DefaultDict, Iterator, Type
from collections import defaultdict

import pathos

if TYPE_CHECKING:
    from blackopt.abc import Solver
    from ilya_ezplot import Metric


class SolverFactory:
    def __init__(self, target_cls: Type['Solver'], *args, **kwargs):
        self.target_cls = target_cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.target_cls(*self.args, **self.kwargs)


def one_trial(steps: int, solver_constructor: SolverFactory) -> DefaultDict[str, 'Metric']:
    s: Solver = solver_constructor()
    print(s)
    s.solve(steps)
    return s.metrics


def n_runs(trials: int, steps: int, solver: SolverFactory) -> 'Metric':

    pool = pathos.pools.ProcessPool()
    metrics = pool.map(lambda x: one_trial(steps, solver), "x" * trials)

    return sum(metrics)


def compare_solvers(
    trials: int, steps: int, solvers: List[SolverFactory], single_process: bool = True
) -> Dict[SolverFactory, Dict[str, 'Metric']]:

    to_map = solvers * trials
    mapping = lambda solver: one_trial(steps, solver)
    if single_process:
        metrics = map(mapping, to_map)
    else:
        pool = pathos.pools.ProcessPool()
        metrics: Iterator[Dict[str, 'Metric']] = pool.map(
            mapping, to_map
        )
    solver_to_metrics = defaultdict(lambda :defaultdict(list))
    for sf, ms in zip(to_map, metrics):
        for key, metric in ms.items():
            solver_to_metrics[sf][key].append(metric)

    result = defaultdict(dict)
    for sf, metrics_dict in solver_to_metrics.items():
        for key, lst in metrics_dict.items():
            result[sf][key] = sum(lst)

    return result
