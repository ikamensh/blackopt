from typing import Callable, List, NoReturn, Optional
from blackopt.abc import Solver

from blackopt.log import get_logger
from blackopt.exceptions import BlackoptException


class ContinuousOptimizer:
    def __init__(
        self,
        solver: Solver,
        evals_per_step=1_000_000,
        step_callbacks: List[Callable[['ContinuousOptimizer'], NoReturn]]= None,
        max_repeats = 5,
        termination_condition: Optional[Callable[['ContinuousOptimizer'], bool]] = None
    ):
        self.solver = solver
        self.problem = solver.problem
        self.evals_per_step = evals_per_step
        self.step_callbacks = step_callbacks or []
        self.max_repeats = max_repeats
        self.termination_condition = termination_condition

        self.logger = get_logger()
        self.steps = 0
        self.repeats = 0
        self.last_exception = "No exception so far"


    def run(self):
        self.logger.info("starting optimization")
        while True:
            try:
                if self.termination_condition and self.termination_condition(self):
                    self.logger.info("Termination condition fulfilled. Exiting.")
                    return

                self.checkpoint()
                self.step()
                self.repeats = 0
            except Exception as e:
                if str(e) == self.last_exception:
                    self.repeats += 1
                    if self.repeats > self.max_repeats:
                        raise BlackoptException(f"Exception reoccured {self.repeats} times") from e
                else:
                    self.last_exception = str(e)
                    self.repeats = 0
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
