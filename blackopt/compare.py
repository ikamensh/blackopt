from typing import List, TYPE_CHECKING, ClassVar, Dict, DefaultDict
from collections import defaultdict
import multiprocessing


if TYPE_CHECKING:
    from blackopt.abc import Solver
    from ilya_ezplot import Metric


class SolverFactory:
    def __init__(self, target_cls: ClassVar['Solver'], *args, **kwargs):
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


# def n_runs(trials: int, steps: int, solver: SolverFactory) -> 'Metric':
#
#     pool = multiprocessing.Pool()
#     metrics = pool.map(lambda x: one_trial(steps, solver), "x" * trials)
#
#     return sum(metrics)


def inner( x ):
    steps, solver = x
    return one_trial(steps, solver)

def inner_defaultdict():
    return defaultdict(list)

def compare_solvers(
    trials: int, steps: int, solvers: List[SolverFactory]
) -> Dict[SolverFactory, Dict[str, 'Metric']]:
    pool = multiprocessing.Pool()
    to_map = [(steps, s) for s in solvers] * trials
    print(f"Using process pool with {pool._processes} CPUs")

    metrics: List[Dict[str, 'Metric']] = pool.map(
        inner, to_map
    )
    solver_to_metrics = defaultdict(inner_defaultdict)
    for sf, ms in zip(to_map, metrics):
        for key, metric in ms.items():
            solver_to_metrics[sf][key].append(metric)

    result = defaultdict(dict)
    for sf, metrics_dict in solver_to_metrics.items():
        for key, lst in metrics_dict.items():
            result[sf][key] = sum(lst)

    return result


