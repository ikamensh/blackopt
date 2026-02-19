from typing import List, Type

from blackopt.abc import Problem, Solution
from blackopt.algorithms import EvolutionaryAlgorithm
from blackopt.exceptions import EarlyStopException


def keep(child_score: float, parent_min: float, diff: float, pressure: float):
    return child_score > parent_min + pressure * diff


class OffspringSelection(EvolutionaryAlgorithm):
    """Evolutionary algorithm with offspring selection. """
    name = "OffspringSelection"

    # Default assumed total evaluations when ``step()`` is used without
    # ``solve()`` (e.g. inside *autosolve*).  This keeps the selection
    # pressure ramp in a sensible range instead of growing unboundedly.
    _DEFAULT_STEPS_ESTIMATE: int = 10_000

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        popsize: int,
        mutation_rate: float,
        elite_size: int = 0,
        equal_chances: float = 0.5,
        max_selective_pressure=200,
        early_stop=True,
    ):
        super().__init__(
            problem, solution_cls, popsize, mutation_rate, elite_size, equal_chances
        )
        self.max_selective_pressure = max_selective_pressure
        self.early_stop = early_stop
        self.selective_pressure = 0
        self._total_steps = None

    def check_early_stop(self):
        return self.selective_pressure >= self.max_selective_pressure

    def solve(self, steps):
        self._total_steps = steps
        super().solve(steps)

    def step(self):
        total = self._total_steps or self._DEFAULT_STEPS_ESTIMATE
        pressure = min(0.8, 0.8 * self.problem.eval_count / total)

        next_generation = self.population[: self.elite_size]
        breed = self._breed(self.popsize - self.elite_size, pressure=pressure)
        if not breed:
            raise EarlyStopException()

        next_generation += breed
        self.population = next_generation

        self._rank()
        self.record()
        self.generation += 1
        if not self.generation % 10:
            print("Generation", self.generation, self.problem.eval_count)

        if self.early_stop and self.check_early_stop():
            raise EarlyStopException()

    def record(self):
        super().record()
        self.record_metric("selective pressure", self.selective_pressure)

    def _breed(self, n: int, pressure=0.5) -> List[Solution]:

        result: List[Solution] = []
        ctr = 0
        while len(result) < n and ctr < self.max_selective_pressure:
            ctr += 1
            # Only request as many parents as we still need children,
            # avoiding wasted crossover/mutation/evaluation work.
            needed = n - len(result)
            children = []
            parents = self._select_parents(needed)

            parent_scores = {}  # child -> min_parent_score, parents_diff
            for i in range(needed):
                parent_1 = parents[i]
                parent_2 = parents[len(parents) - i - 1]
                new = parent_1.crossover(parent_2)
                mutated = []
                for c in new:
                    c = c.mutate(self.mutation_rate)
                    mutated.append(c)
                    parent_scores[c] = (
                        min([parent_1.score, parent_2.score]),
                        abs(parent_1.score - parent_2.score),
                    )

                children += mutated

            result += [
                c for c in children if keep(c.score, *parent_scores[c], pressure)
            ]

        self.selective_pressure = ctr

        return result[:n]
