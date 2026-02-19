import random
from typing import List, Type

from blackopt.abc import Problem, Solution, Solver


class TruncationSelectionGA(Solver):
    """Truncation-Selection Genetic Algorithm.

    This implementation uses a **truncation-selection** strategy rather than
    fitting an explicit probabilistic distribution model.  Each generation:

    1. The population is ranked by fitness.
    2. The top-*k* individuals are kept as the *elite*.
    3. New offspring are created by **crossover** (and optional mutation)
       between randomly sampled elite pairs, until the population is full.

    The approach captures the *spirit* of EDA — biasing reproduction toward
    high-fitness regions — while remaining agnostic to the solution
    representation (it works with any ``Solution`` subclass that implements
    ``crossover`` and ``mutate``).
    """

    name = "TruncationSelectionGA"

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        popsize: int,
        top_k: int,
        mutation_rate: float = 0.0,
    ):
        assert popsize > 1
        assert 1 <= top_k < popsize

        self.popsize = popsize
        self.top_k = top_k
        self.mutation_rate = mutation_rate

        super().__init__(problem, solution_cls)

        self.population = [solution_cls.random_solution() for _ in range(popsize)]
        self._rank()

    def _rank(self):
        self.population.sort(
            key=lambda x: x.score + x.regularization_score, reverse=True
        )
        self.best_solution = max(
            self.population + [self.best_solution],
            key=lambda x: x.score,
        )

    def _breed(self, n: int) -> List[Solution]:
        elite = self.population[: self.top_k]
        children = []
        while len(children) < n:
            p1, p2 = random.sample(elite, 2)
            offspring = p1.crossover(p2)
            for child in offspring:
                if len(children) >= n:
                    break
                if self.mutation_rate > 0:
                    child = child.mutate(self.mutation_rate)
                children.append(child)
        return children

    def step(self):
        elite = self.population[: self.top_k]
        children = self._breed(self.popsize - self.top_k)
        self.population = elite + children
        self._rank()
        self.record()

    def __str__(self):
        return (
            f"{self.name} pop-{self.popsize} top-{self.top_k} "
            f"mut-{self.mutation_rate}"
        )


# Backward-compatible alias: kept for compatibility but this is actually a
# Truncation Selection GA, not an Estimation of Distribution Algorithm.
Eda = TruncationSelectionGA
