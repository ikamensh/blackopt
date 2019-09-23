import os
import datetime
from typing import List

from ilya_ezplot import Metric, plot_group

from config import root_dir
from blackbox.abc import Problem


def generate_report(problem: Problem, metrics: List[Metric]):
    """ plot multiple curves from the metrics list"""
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    problem_path = os.path.join(root_dir, "reports", str(problem))
    os.makedirs(problem_path, exist_ok=True)
    plot_group(metrics, problem_path, name=f"Best scores @ {timestamp}")





