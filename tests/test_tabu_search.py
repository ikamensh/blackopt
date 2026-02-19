"""Tests for TabuSearch algorithm."""

import pytest

from blackopt.algorithms import TabuSearch
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.examples.problems.bumpy import BumpyProblem, BumpySolution


# ---- helpers ---------------------------------------------------------------

def _step_problem(n_dim: int = 20):
    return StepProblem.random_problem(n_dim)


def _bumpy_problem(n_dim: int = 5, n_expr: int = 10):
    return BumpyProblem.random_problem(n_dim, n_expr)


# ---- tests -----------------------------------------------------------------

def test_tabu_basic_step():
    """TabuSearch can make a step and maintains best_solution."""
    problem = _step_problem()
    ts = TabuSearch(problem, StepSolution, mutation_rate=0.1)
    initial_best = ts.best_solution.score

    for _ in range(50):
        ts.step()

    assert ts.best_solution.score >= initial_best


def test_tabu_current_solution_updates():
    """current_solution should differ from the initial one after several steps."""
    problem = _step_problem(n_dim=10)
    ts = TabuSearch(problem, StepSolution, mutation_rate=0.3)
    initial_current = ts.current_solution

    for _ in range(20):
        ts.step()

    # It's extremely unlikely that the walker stays at the exact same solution
    # for 20 steps with a 30% mutation rate.
    assert ts.current_solution is not initial_current


def test_tabu_list_fills():
    """The tabu list should grow as steps are taken (up to tabu_size)."""
    problem = _step_problem(n_dim=10)
    tabu_size = 5
    ts = TabuSearch(
        problem, StepSolution, mutation_rate=0.2, tabu_size=tabu_size,
    )

    assert len(ts.tabu_list) == 0

    for _ in range(20):
        ts.step()

    # After enough steps the tabu list should be full.
    assert len(ts.tabu_list) == tabu_size


def test_tabu_with_bumpy_problem():
    """TabuSearch works on a different problem type (BumpyProblem)."""
    problem = _bumpy_problem()
    ts = TabuSearch(problem, BumpySolution, mutation_rate=0.3)
    initial_best = ts.best_solution.score

    for _ in range(30):
        ts.step()

    assert ts.best_solution.score >= initial_best


def test_tabu_custom_params():
    """Custom tabu_threshold and n_candidates are respected."""
    problem = _step_problem(n_dim=10)
    ts = TabuSearch(
        problem, StepSolution,
        mutation_rate=0.2,
        tabu_size=3,
        tabu_threshold=0.5,
        n_candidates=10,
    )

    assert ts.tabu_size == 3
    assert ts.tabu_threshold == 0.5
    assert ts.n_candidates == 10

    for _ in range(10):
        ts.step()

    assert ts.best_solution.score >= 0


def test_tabu_str():
    """__str__ should include the algorithm name and parameters."""
    problem = _step_problem()
    ts = TabuSearch(problem, StepSolution, mutation_rate=0.1, tabu_size=7)

    s = str(ts)
    assert "Tabu" in s
    assert "0.1" in s
    assert "7" in s
