import math
import random
from typing import Type

from blackopt.abc import Problem, Solution, Solver


class SimAnneal(Solver):
    name = "Simulated Annealing"
    # On rejection, the walker stays at current_solution (random-walk behaviour)
    # rather than resetting to the global best.  The default cooling schedule
    # assumes ~1M steps when ``step()`` is used without ``solve()`` (e.g. in
    # autosolve), keeping temperature high across many rounds.
    _DEFAULT_STEPS_ESTIMATE: int = 1_000_000

    def __init__(self, problem: Problem, solution_cls: Type[Solution], mutation_rate, temp=0.5, half_times=20):

        super().__init__(problem, solution_cls)

        self.mutation_rate = mutation_rate
        self.temp_initial = temp
        self.temp = temp
        self.half_times = half_times
        # alpha is set by solve(); when step() is used directly (e.g. via
        # autosolve) _ensure_alpha() will initialise it on the first call.
        self.alpha = None
        self.current_solution = self.best_solution


    @property
    def cur_fitness(self):
        return self.current_solution.score

    @property
    def best_fitness(self):
        return self.best_solution.score


    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        try:
            return math.exp(
                -abs(candidate_fitness - self.cur_fitness)
                / (self.problem.score_span * self.temp)
            )
        except (OverflowError, ZeroDivisionError):
            # temp ≈ 0 → denominator ≈ 0 → exponent → −∞ → accept prob → 0
            return 0.0

    def maybe_accept(self, candidate: Solution):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probability p_accept(..) if candidate is worse.
        On rejection the walker stays at current_solution (standard SA
        random-walk behaviour) rather than resetting to the global best.
        """
        candidate_fitness = candidate.score
        if candidate_fitness > self.cur_fitness:
            self.current_solution = candidate
            if candidate_fitness > self.best_fitness:
                self.best_solution = candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.current_solution = candidate
            # else: stay at self.current_solution (do nothing)

    def _ensure_alpha(self, steps_estimate: int | None = None) -> None:
        """Initialise ``alpha`` if it has not been set yet.

        Called automatically on the first ``step()`` invocation when
        ``solve()`` was never called (e.g. inside *autosolve*).
        """
        if self.alpha is not None:
            return
        steps = steps_estimate or self._DEFAULT_STEPS_ESTIMATE
        self.alpha = 2 ** (- self.half_times / steps)

    def solve(self, steps):
        """
        Execute simulated annealing algorithm.
        Initializes temperature schedule, then delegates to base Solver.solve().
        """
        self.temp = self.temp_initial
        self.alpha = 2 ** (- self.half_times / steps)
        super().solve(steps)

    def step(self):
        self._ensure_alpha()
        candidate = self.current_solution.mutate(self.mutation_rate)
        self.maybe_accept(candidate)
        self.temp *= self.alpha
        self.record()

    def __str__(self):
        return f"{self.name} mr-{self.mutation_rate} ht-{self.half_times} t-{self.temp_initial}"


