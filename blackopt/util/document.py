import os
import datetime
from typing import Dict
from collections import defaultdict

from blackopt.util.metric import Metric
from blackopt.compare import SolverFactory
from blackopt.abc import Problem
from blackopt.config import get_rootdir


def generate_report(problem: Problem, metrics: Dict[SolverFactory, Dict[str, Metric]]):
    """Plot multiple curves from the metrics list. Plotting to be implemented."""
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    problem_path = os.path.join(get_rootdir(), "reports", str(problem))

    m_groups = defaultdict(list)

    for sf, ms_dict in metrics.items():
        for key, m in ms_dict.items():
            m.discard_warmup(0.15)
            m_groups[key].append(m)

    # TODO: add plotting (plot_group replacement)
