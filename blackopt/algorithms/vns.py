from typing import Type

from blackopt.abc import Problem, Solution, Solver


class VariableNeighborhoodSearch(Solver):
    """Variable Neighborhood Search: tries increasing mutation rates to escape
    local optima; resets to minimum rate on improvement; optionally restarts."""

    name = "VNS"

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        mutation_rates: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
        attempts_per_rate: int = 3,
        random_restart: bool = True,
    ):
        assert len(mutation_rates) >= 1
        assert all(0 < r <= 1 for r in mutation_rates)
        assert mutation_rates == tuple(sorted(mutation_rates))
        assert attempts_per_rate >= 1

        self.mutation_rates = mutation_rates
        self.attempts_per_rate = attempts_per_rate
        self.random_restart = random_restart

        super().__init__(problem, solution_cls)

        self.current_solution = self.best_solution

    def step(self):
        improved = False
        for rate in self.mutation_rates:
            if improved:
                break
            for _ in range(self.attempts_per_rate):
                candidate = self.current_solution.mutate(rate)
                if candidate.score > self.current_solution.score:
                    self.current_solution = candidate
                    if candidate.score > self.best_solution.score:
                        self.best_solution = candidate
                    improved = True
                    break

        if not improved and self.random_restart:
            self.current_solution = self.solution_cls.random_solution()
            if self.current_solution.score > self.best_solution.score:
                self.best_solution = self.current_solution

        self.record()

    def __str__(self):
        return (
            f"{self.name} rates-{self.mutation_rates} "
            f"attempts-{self.attempts_per_rate} restart-{self.random_restart}"
        )
