"""Particle Swarm Optimization (PSO).

Operates directly on solutions that expose a ``genes`` attribute (a list of
floats) and whose constructor accepts a single ``genes`` argument.  If the
solution class does not expose ``genes``, a ``TypeError`` is raised at
construction time so the caller can fall back to another algorithm.

Each particle maintains a *position* (the gene vector), a *velocity*, and
a *personal best* position.  On every step the velocity is updated using
the classic PSO rule::

    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)

where *w* is the inertia weight, *c1* and *c2* are the cognitive and social
acceleration constants, and *r1*, *r2* are independent uniform random numbers
in [0, 1].  Positions are clamped to [0, 1] after each update.
"""

import random
from typing import List, Type

from blackopt.abc import Problem, Solution
from blackopt.abc.solver import Solver


class _Particle:
    """Internal bookkeeping for a single PSO particle."""

    __slots__ = ("position", "velocity", "pbest_position", "pbest_score")

    def __init__(self, position: List[float]):
        self.position: List[float] = list(position)
        self.velocity: List[float] = [0.0] * len(position)
        self.pbest_position: List[float] = list(position)
        self.pbest_score: float = float("-inf")


class ParticleSwarmOptimization(Solver):
    """Classic Particle Swarm Optimization.

    Parameters
    ----------
    problem : Problem
        The optimisation problem.
    solution_cls : type
        A concrete :class:`Solution` subclass whose instances have a
        ``genes`` attribute (``list[float]``) and whose constructor
        accepts a positional ``genes`` argument.
    popsize : int
        Swarm size (number of particles).
    w : float
        Inertia weight, typically in [0.4, 0.9].  Controls how much of the
        previous velocity is retained.
    c1 : float
        Cognitive acceleration constant (personal best attraction).
    c2 : float
        Social acceleration constant (global best attraction).
    """

    name = "PSO"

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        popsize: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        assert popsize >= 2, "PSO needs at least 2 particles"
        assert 0 <= w <= 1, "Inertia w should be in [0, 1]"
        assert c1 >= 0, "c1 must be non-negative"
        assert c2 >= 0, "c2 must be non-negative"

        self.popsize = popsize
        self.w = w
        self.c1 = c1
        self.c2 = c2

        super().__init__(problem, solution_cls)

        # Build the initial swarm and validate that genes are available.
        self.population: List[Solution] = []
        self._particles: List[_Particle] = []

        for _ in range(popsize):
            ind = solution_cls.random_solution()
            if not hasattr(ind, "genes"):
                raise TypeError(
                    f"{solution_cls.__name__} instances do not expose a "
                    f"'genes' attribute.  ParticleSwarmOptimization requires "
                    f"real-valued gene vectors."
                )
            self.population.append(ind)
            self._particles.append(_Particle(ind.genes))

        # Evaluate initial personal bests and global best.
        for sol, particle in zip(self.population, self._particles):
            score = sol.score
            particle.pbest_score = score
            particle.pbest_position = list(sol.genes)

        self.best_solution = max(
            self.population + [self.best_solution],
            key=lambda s: s.score,
        )

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp *value* into [lo, hi]."""
        return max(lo, min(hi, value))

    def step(self):
        """One PSO iteration: update velocities, positions, and bests."""
        gbest = self.best_solution.genes

        next_pop: List[Solution] = []

        for idx in range(self.popsize):
            particle = self._particles[idx]
            pos = particle.position
            vel = particle.velocity
            pbest = particle.pbest_position
            n_dim = len(pos)

            # Update velocity and position.
            new_vel = [0.0] * n_dim
            new_pos = [0.0] * n_dim
            for j in range(n_dim):
                r1 = random.random()
                r2 = random.random()
                new_vel[j] = (
                    self.w * vel[j]
                    + self.c1 * r1 * (pbest[j] - pos[j])
                    + self.c2 * r2 * (gbest[j] - pos[j])
                )
                new_pos[j] = self._clip(pos[j] + new_vel[j])

            particle.velocity = new_vel
            particle.position = new_pos

            # Create a new solution from the updated position.
            sol = self.solution_cls(list(new_pos))
            score = sol.score

            # Update personal best.
            if score > particle.pbest_score:
                particle.pbest_score = score
                particle.pbest_position = list(new_pos)

            next_pop.append(sol)

        self.population = next_pop

        # Update global best.
        self.best_solution = max(
            self.population + [self.best_solution],
            key=lambda s: s.score,
        )

        self.record()

    def __str__(self):
        return f"{self.name} pop-{self.popsize} w-{self.w} c1-{self.c1} c2-{self.c2}"
