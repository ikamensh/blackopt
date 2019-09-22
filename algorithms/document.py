from problems import Problem
import os
import datetime
from typing import List

from ilya_ezplot import Metric, plot_group

from config import root_dir


class PlotProgress:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        self.problem_path = os.path.join(root_dir, "reports", str(self.problem))
        self.create_report_folders()

    def create_report_folders(self):
        if not os.path.isdir(self.problem_path):
            os.makedirs(self.problem_path)

    def generate_report(self, metrics: List[Metric]):
        """ plot multiple curves from the metrics list"""
        plot_group(metrics, self.problem_path, name=f"Best scores @ {self.timestamp}")
