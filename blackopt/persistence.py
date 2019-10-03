from __future__ import annotations

from typing import Callable, List, NoReturn
from blackopt.abc import Solver

from blackopt.log import get_logger


class ContinuousOptimizer:
    def __init__(
        self,
        problem,
        solver_factory: Callable[[], Solver],
        evals_per_step=1_000_000,
        step_callbacks: List[Callable[[ContinuousOptimizer], NoReturn]]= None,
    ):
        self.problem = problem
        self.evals_per_step = evals_per_step
        self.logger = get_logger()
        self.steps = 0
        self.step_callbacks = step_callbacks or []

        self.solver = solver_factory()

    def run(self):
        self.logger.info("starting optimization")
        while True:
            try:
                self.step()
                self.checkpoint()
            except Exception as e:
                self.logger.warning(f"Encountered an exception: {e}")
                self.restore_latest()

    def step(self):
        for callback in self.step_callbacks:
            callback(self)

        self.solver.solve(self.evals_per_step)
        self.steps += 1
        self.logger.info(
            "step done", step=self.steps, score=self.solver.best_solution.score
        )

    def checkpoint(self):
        self.solver.checkpoint()

    def restore_latest(self):
        self.solver = Solver.restore_latest(self.problem)
