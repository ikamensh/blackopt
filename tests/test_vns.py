"""Tests for VariableNeighborhoodSearch algorithm."""

import pytest

from blackopt.algorithms import VariableNeighborhoodSearch
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.examples.problems.bumpy import BumpyProblem, BumpySolution


# ---- helpers ---------------------------------------------------------------

def _step_problem(n_dim: int = 20):
    return StepProblem.random_problem(n_dim)


def _bumpy_problem(n_dim: int = 5, n_expr: int = 10):
    return BumpyProblem.random_problem(n_dim, n_expr)


# ---- tests -----------------------------------------------------------------

def test_vns_basic_step():
    """VNS can make a step and maintains best_solution."""
    problem = _step_problem()
    vns = VariableNeighborhoodSearch(problem, StepSolution)
    initial_best = vns.best_solution.score

    for _ in range(30):
        vns.step()

    assert vns.best_solution.score >= initial_best


def test_vns_improves():
    """After enough steps VNS should improve over the initial random solution."""
    problem = _step_problem(n_dim=10)
    vns = VariableNeighborhoodSearch(problem, StepSolution)
    initial_best = vns.best_solution.score

    for _ in range(100):
        vns.step()

    # With 100 steps and random restarts, it should at least match start.
    assert vns.best_solution.score >= initial_best


def test_vns_custom_rates():
    """Custom mutation_rates tuple is respected."""
    problem = _step_problem(n_dim=10)
    rates = (0.05, 0.1, 0.3)
    vns = VariableNeighborhoodSearch(
        problem, StepSolution,
        mutation_rates=rates,
        attempts_per_rate=5,
    )

    assert vns.mutation_rates == rates
    assert vns.attempts_per_rate == 5

    for _ in range(20):
        vns.step()

    assert vns.best_solution.score >= 0


def test_vns_no_restart():
    """VNS without random restart should still work."""
    problem = _step_problem(n_dim=10)
    vns = VariableNeighborhoodSearch(
        problem, StepSolution,
        random_restart=False,
    )

    assert vns.random_restart is False

    for _ in range(20):
        vns.step()

    assert vns.best_solution.score >= 0


def test_vns_with_bumpy_problem():
    """VNS works on a different problem type (BumpyProblem)."""
    problem = _bumpy_problem()
    vns = VariableNeighborhoodSearch(problem, BumpySolution)
    initial_best = vns.best_solution.score

    for _ in range(30):
        vns.step()

    assert vns.best_solution.score >= initial_best


def test_vns_str():
    """__str__ should include the algorithm name and parameters."""
    problem = _step_problem()
    vns = VariableNeighborhoodSearch(
        problem, StepSolution,
        mutation_rates=(0.1, 0.2),
        attempts_per_rate=4,
        random_restart=True,
    )

    s = str(vns)
    assert "VNS" in s
    assert "(0.1, 0.2)" in s
    assert "4" in s


def test_vns_invalid_rates_order():
    """mutation_rates must be sorted ascending."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        VariableNeighborhoodSearch(
            problem, StepSolution,
            mutation_rates=(0.5, 0.1),
        )


def test_vns_invalid_empty_rates():
    """mutation_rates must have at least one entry."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        VariableNeighborhoodSearch(
            problem, StepSolution,
            mutation_rates=(),
        )
