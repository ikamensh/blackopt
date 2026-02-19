"""Tests for TruncationSelectionGA (formerly EDA)."""

import pytest

from blackopt.algorithms import TruncationSelectionGA, Eda
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.examples.problems.bumpy import BumpyProblem, BumpySolution


# ---- helpers ---------------------------------------------------------------

def _step_problem(n_dim: int = 20):
    return StepProblem.random_problem(n_dim)


def _bumpy_problem(n_dim: int = 5, n_expr: int = 10):
    return BumpyProblem.random_problem(n_dim, n_expr)


# ---- tests -----------------------------------------------------------------

def test_backward_compat_alias():
    """Eda is a backward-compatible alias for TruncationSelectionGA."""
    assert Eda is TruncationSelectionGA


def test_tsga_basic_step():
    """TruncationSelectionGA can make a step and maintains best_solution."""
    problem = _step_problem()
    tsga = TruncationSelectionGA(problem, StepSolution, popsize=10, top_k=3)
    initial_best = tsga.best_solution.score

    for _ in range(20):
        tsga.step()

    assert tsga.best_solution.score >= initial_best


def test_tsga_population_size():
    """Population is maintained at exactly the requested size."""
    problem = _step_problem(n_dim=10)
    pop = 12
    tsga = TruncationSelectionGA(problem, StepSolution, popsize=pop, top_k=4)

    assert len(tsga.population) == pop
    tsga.step()
    assert len(tsga.population) == pop


def test_tsga_with_mutation():
    """TruncationSelectionGA with a positive mutation rate should still work."""
    problem = _step_problem(n_dim=10)
    tsga = TruncationSelectionGA(
        problem, StepSolution,
        popsize=8, top_k=3, mutation_rate=0.1,
    )
    initial_best = tsga.best_solution.score

    for _ in range(30):
        tsga.step()

    assert tsga.best_solution.score >= initial_best


def test_tsga_without_mutation():
    """TruncationSelectionGA with mutation_rate=0 still works (pure crossover)."""
    problem = _step_problem(n_dim=10)
    tsga = TruncationSelectionGA(
        problem, StepSolution,
        popsize=8, top_k=3, mutation_rate=0.0,
    )

    for _ in range(10):
        tsga.step()

    assert tsga.best_solution.score >= 0


def test_tsga_with_bumpy_problem():
    """TruncationSelectionGA works on a different problem type (BumpyProblem)."""
    problem = _bumpy_problem()
    tsga = TruncationSelectionGA(problem, BumpySolution, popsize=10, top_k=3)
    initial_best = tsga.best_solution.score

    for _ in range(20):
        tsga.step()

    assert tsga.best_solution.score >= initial_best


def test_tsga_ranking():
    """After a step the population should be sorted by score (descending)."""
    problem = _step_problem(n_dim=10)
    tsga = TruncationSelectionGA(problem, StepSolution, popsize=10, top_k=3)

    tsga.step()

    scores = [s.score + s.regularization_score for s in tsga.population]
    assert scores == sorted(scores, reverse=True)


def test_tsga_str():
    """__str__ should include the algorithm name and parameters."""
    problem = _step_problem()
    tsga = TruncationSelectionGA(problem, StepSolution, popsize=15, top_k=5, mutation_rate=0.2)

    s = str(tsga)
    assert "TruncationSelectionGA" in s
    assert "15" in s
    assert "5" in s


def test_tsga_invalid_top_k():
    """top_k must be >= 1 and < popsize."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        TruncationSelectionGA(problem, StepSolution, popsize=5, top_k=0)

    with pytest.raises(AssertionError):
        TruncationSelectionGA(problem, StepSolution, popsize=5, top_k=5)


def test_tsga_invalid_popsize():
    """popsize must be > 1."""
    problem = _step_problem()
    with pytest.raises(AssertionError):
        TruncationSelectionGA(problem, StepSolution, popsize=1, top_k=1)
