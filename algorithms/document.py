from algorithms.solver import Solver
from problems import Problem
import datetime
import os
from config import root_dir
from ilya_ezplot import Metric, plot_group
from typing import Dict


class PlotProgress:

    def __init__(self, problem: Problem, solver: Solver = None):
        self.problem = problem
        self.timestamp = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')

        self.problem_path = os.path.join(
            root_dir, "reports", str(self.problem))

        self.solver = solver
        if solver:
            self.solver_path = os.path.join(
                self.problem_path, self.solver.name, self.timestamp)

        self.create_report_folders()

    def create_report_folders(self):
        if self.solver:
            if not os.path.isdir(self.solver_path):
                os.makedirs(self.solver_path)
        else:
            if not os.path.isdir(self.problem_path):
                os.makedirs(self.problem_path)

    def generate_report(self, metrics: Dict[str, Metric]):
        """ plot multiple curves from the metrics dict [label , Metric]"""
        if self.solver:
            plot_group(
                metrics,
                self.solver_path,
                name=f"Best scores @ {self.timestamp}")
        else:
            plot_group(
                metrics,
                self.problem_path,
                name=f"Best scores @ {self.timestamp}")
