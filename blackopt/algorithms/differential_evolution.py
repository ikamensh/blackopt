"""Differential Evolution (DE/rand/1/bin).

Operates directly on solutions that expose a ``genes`` attribute (a list of
floats) and whose constructor accepts a single ``genes`` argument.  If the
solution class does not expose ``genes``, a ``TypeError`` is raised at
construction time so the caller can fall back to another algorithm.
"""

import random
from typing import List, Type

from blackopt.abc import Problem, Solution
from blackopt.abc.solver import Solver


class DifferentialEvolution(Solver):
    """Classic DE/rand/1/bin strategy.

    Parameters
    ----------
    problem : Problem
        The optimisation problem.
    solution_cls : type
        A concrete :class:`Solution` subclass whose instances have a
        ``genes`` attribute (``list[float]``) and whose constructor
        accepts a positional ``genes`` argument.
    popsize : int
        Population size (>= 4 for meaningful donor selection).
    F : float
        Differential weight / scale factor, typically in [0.4, 1.0].
    CR : float
        Crossover probability, typically in [0.5, 1.0].
    """

    name = "DE"

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        popsize: int = 20,
        F: float = 0.8,
        CR: float = 0.9,
    ):
        assert popsize >= 4, "DE needs at least 4 individuals"
        assert 0 < F <= 2, "F should be in (0, 2]"
        assert 0 <= CR <= 1, "CR should be in [0, 1]"

        self.popsize = popsize
        self.F = F
        self.CR = CR

        super().__init__(problem, solution_cls)

        # Build the initial population and validate that genes are available.
        self.population: List[Solution] = []
        for _ in range(popsize):
            ind = solution_cls.random_solution()
            if not hasattr(ind, "genes"):
                raise TypeError(
                    f"{solution_cls.__name__} instances do not expose a "
                    f"'genes' attribute.  DifferentialEvolution requires "
                    f"real-valued gene vectors."
                )
            self.population.append(ind)

        self._rank()

    # -- internals ------------------------------------------------------------

    def _rank(self):
        self.population.sort(
            key=lambda s: s.score + s.regularization_score, reverse=True
        )
        self.best_solution = max(
            self.population + [self.best_solution],
            key=lambda s: s.score,
        )

    @staticmethod
    def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp *value* into [lo, hi]."""
        return max(lo, min(hi, value))

    def _trial_vector(self, target_idx: int) -> List[float]:
        """Create a trial vector via DE/rand/1/bin."""
        # Pick three distinct individuals, all different from target.
        idxs = list(range(self.popsize))
        idxs.remove(target_idx)
        a_idx, b_idx, c_idx = random.sample(idxs, 3)

        a = self.population[a_idx].genes
        b = self.population[b_idx].genes
        c = self.population[c_idx].genes
        target = self.population[target_idx].genes

        n_dim = len(target)
        # Ensure at least one dimension is taken from the donor.
        j_rand = random.randrange(n_dim)

        trial = []
        for j in range(n_dim):
            if random.random() < self.CR or j == j_rand:
                trial.append(self._clip(a[j] + self.F * (b[j] - c[j])))
            else:
                trial.append(target[j])
        return trial

    # -- public Solver interface ----------------------------------------------

    def step(self):
        """One generational step: produce a trial for every population member
        and keep it if it is at least as good as the target."""
        next_pop: List[Solution] = []
        for i in range(self.popsize):
            trial_genes = self._trial_vector(i)
            trial = self.solution_cls(trial_genes)

            if trial.score >= self.population[i].score:
                next_pop.append(trial)
            else:
                next_pop.append(self.population[i])

        self.population = next_pop
        self._rank()
        self.record()

    def __str__(self):
        return f"{self.name} pop-{self.popsize} F-{self.F} CR-{self.CR}"
