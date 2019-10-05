import os
import datetime
from typing import Dict
from collections import defaultdict

from ilya_ezplot import Metric, plot_group

from blackopt.compare import SolverFactory
from blackopt.abc import Problem
from blackopt.config import get_rootdir


def generate_report(problem: Problem, metrics: Dict[SolverFactory, Dict[str, Metric]]):
    """ plot multiple curves from the metrics list"""
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    problem_path = os.path.join(get_rootdir(), "reports", str(problem))

    m_groups = defaultdict(list)

    for sf, ms_dict in metrics.items():
        for key, m in ms_dict.items():
            m.discard_warmup(0.15)
            m_groups[key].append(m)

    for key, ms in m_groups.items():
        plot_group(
            ms,
            f"{problem_path}@{timestamp}",
            name=key,
            stdev_factor=0.1,
            smoothen=False,
        )
