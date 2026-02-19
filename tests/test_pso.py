"""Tests for Particle Swarm Optimization (PSO)."""

import copy

import pytest

from blackopt.algorithms import ParticleSwarmOptimization
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.examples.problems.bumpy import BumpyProblem, BumpySolution


# ---- helpers ---------------------------------------------------------------

def _step_problem(n_dim: int = 20):
    return StepProblem.random_problem(n_dim)


def _bumpy_problem(n_dim: int = 5, n_expr: int = 10):
    return BumpyProblem.random_problem(n_dim, n_expr)


# ---- tests -----------------------------------------------------------------

def test_pso_basic_step():
    """PSO can make a step and maintains best_solution."""
    problem = _step_problem()
    pso = ParticleSwarmOptimization(problem, StepSolution, popsize=10)
    initial_best = pso.best_solution.score

    for _ in range(20):
        pso.step()

    assert pso.best_solution.score >= initial_best


def test_pso_population_size():
    """Population is maintained at exactly the requested size."""
    problem = _step_problem(n_dim=10)
    pop = 12
    pso = ParticleSwarmOptimization(problem, StepSolution, popsize=pop)

    assert len(pso.population) == pop
    pso.step()
    assert len(pso.population) == pop


def test_pso_particle_state():
    """Internal particle state should be consistent with the population."""
    problem = _step_problem(n_dim=10)
    pop = 8
    pso = ParticleSwarmOptimization(problem, StepSolution, popsize=pop)

    assert len(pso._particles) == pop

    for _ in range(5):
        pso.step()

    # After steps, each particle's position should match its solution's genes.
    for sol, particle in zip(pso.population, pso._particles):
        assert sol.genes == particle.position


def test_pso_personal_best_improves():
    """Personal best scores should never decrease."""
    problem = _step_problem(n_dim=10)
    pso = ParticleSwarmOptimization(problem, StepSolution, popsize=6)

    # Record initial personal bests.
    initial_pbests = [p.pbest_score for p in pso._particles]

    for _ in range(20):
        pso.step()

    for i, particle in enumerate(pso._particles):
        assert particle.pbest_score >= initial_pbests[i]


def test_pso_genes_clipping():
    """Positions should stay in [0, 1] even with aggressive parameters."""
    problem = _step_problem(n_dim=5)
    # High c1/c2 with w=0.9 should push particles aggressively.
    pso = ParticleSwarmOptimization(
        problem, StepSolution,
        popsize=8, w=0.9, c1=3.0, c2=3.0,
    )

    for _ in range(20):
        pso.step()

    for sol in pso.population:
        for g in sol.genes:
            assert 0.0 <= g <= 1.0, f"Gene {g} out of [0, 1]"


def test_pso_with_bumpy_problem():
    """PSO works on a different problem type (BumpyProblem)."""
    problem = _bumpy_problem()
    pso = ParticleSwarmOptimization(problem, BumpySolution, popsize=10)
    initial_best = pso.best_solution.score

    for _ in range(20):
        pso.step()

    assert pso.best_solution.score >= initial_best


def test_pso_custom_params():
    """Custom w, c1, c2 are stored correctly."""
    problem = _step_problem(n_dim=10)
    pso = ParticleSwarmOptimization(
        problem, StepSolution,
        popsize=6, w=0.4, c1=2.0, c2=2.0,
    )

    assert pso.w == 0.4
    assert pso.c1 == 2.0
    assert pso.c2 == 2.0
    assert pso.popsize == 6

    for _ in range(10):
        pso.step()

    assert pso.best_solution.score >= 0


def test_pso_str():
    """__str__ should include the algorithm name and parameters."""
    problem = _step_problem()
    pso = ParticleSwarmOptimization(
        problem, StepSolution,
        popsize=15, w=0.7, c1=1.5, c2=1.5,
    )

    s = str(pso)
    assert "PSO" in s
    assert "15" in s
    assert "0.7" in s


def test_pso_no_genes_raises():
    """PSO raises TypeError for solutions without genes."""
    from blackopt.abc import Solution

    class NoGenesSolution(Solution):
        problem = None

        @staticmethod
        def random_solution():
            return NoGenesSolution()

        def mutate(self, rate):
            return copy.deepcopy(self)

        def crossover(self, other):
            return [copy.deepcopy(self)]

    problem = _step_problem()
    NoGenesSolution.problem = problem
    with pytest.raises(TypeError, match="genes"):
        ParticleSwarmOptimization(problem, NoGenesSolution, popsize=4)


def test_pso_invalid_popsize():
    """popsize must be >= 2."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        ParticleSwarmOptimization(problem, StepSolution, popsize=1)


def test_pso_invalid_inertia():
    """Inertia w must be in [0, 1]."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        ParticleSwarmOptimization(problem, StepSolution, popsize=4, w=1.5)
    with pytest.raises(AssertionError):
        ParticleSwarmOptimization(problem, StepSolution, popsize=4, w=-0.1)
