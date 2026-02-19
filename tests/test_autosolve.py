import copy
import time

import dill
import pytest

from blackopt.autosolve import autosolve, Result, _MIN_TIME_SLICE

pytestmark = pytest.mark.slow
from blackopt.algorithms import (
    DifferentialEvolution,
    RandomSearch,
    HillClimber,
    SimAnneal,
)
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.examples.problems.tsp import TspProblem, TspSolution


# ---- helpers ---------------------------------------------------------------

def _make_problem(n_dim: int = 20):
    return StepProblem.random_problem(n_dim)


# ---- tests -----------------------------------------------------------------

def test_autosolve_returns_result():
    """autosolve returns a Result with a scored best_solution and non-empty logs."""
    problem = _make_problem()
    result = autosolve(problem, StepSolution, timeout=3)

    assert isinstance(result, Result)
    assert result.best_solution is not None
    assert isinstance(result.best_solution.score, (int, float))

    # Logs should contain the expected keys.
    assert "convergence" in result.logs
    assert "allocation_history" in result.logs
    assert "winners" in result.logs
    assert "rounds" in result.logs


def test_autosolve_improves_over_random():
    """Given enough time, the portfolio should beat a single random solution."""
    problem = _make_problem()
    StepSolution.problem = problem

    baseline = StepSolution.random_solution().score
    result = autosolve(problem, StepSolution, timeout=3)

    # The portfolio should do at least as well as one random shot.
    assert result.best_solution.score >= baseline


def test_autosolve_respects_timeout():
    """Wall-clock time should stay close to the requested timeout."""
    problem = _make_problem()
    timeout = 3
    t0 = time.monotonic()
    autosolve(problem, StepSolution, timeout=timeout)
    elapsed = time.monotonic() - t0

    # Allow generous slack for process pool overhead but not unbounded.
    assert elapsed < timeout + 3.0


def test_autosolve_custom_algorithms():
    """Passing an explicit list of solvers works."""
    problem = _make_problem()
    solvers = [
        RandomSearch(dill.copy(problem), StepSolution),
        HillClimber(dill.copy(problem), StepSolution, mutation_rate=0.1),
    ]
    # Capture names before the pool might mutate __str__ output.
    names = [str(s) for s in solvers]
    result = autosolve(problem, StepSolution, timeout=3, algorithms=solvers)

    assert isinstance(result, Result)
    assert result.best_solution.score >= 0

    # Convergence logs should have entries for each solver.
    # solver_names now include a " #i" suffix for uniqueness.
    conv_keys = list(result.logs["convergence"].keys())
    assert len(conv_keys) == len(solvers)
    for name in names:
        assert any(name in key for key in conv_keys), (
            f"Expected '{name}' in convergence keys: {conv_keys}"
        )


def test_autosolve_convergence_logs_populated():
    """Each solver should have at least one convergence data point."""
    problem = _make_problem()
    result = autosolve(problem, StepSolution, timeout=3)

    for name, entries in result.logs["convergence"].items():
        assert len(entries) >= 1, f"No convergence data for {name}"
        wall_time, score = entries[0]
        assert wall_time >= 0
        assert isinstance(score, (int, float))


def test_autosolve_winners():
    """Winners should list (round_index, leader_name, leader_score,
    most_improved_name, most_improved_delta) per round."""
    problem = _make_problem()
    result = autosolve(problem, StepSolution, timeout=3)

    winners = result.logs["winners"]
    assert len(winners) == result.logs["rounds"]
    for round_idx, leader_name, leader_score, most_improved_name, most_improved_delta in winners:
        assert isinstance(round_idx, int)
        assert isinstance(leader_name, str)
        assert isinstance(leader_score, (int, float))
        assert isinstance(most_improved_name, str)
        assert isinstance(most_improved_delta, (int, float))


def test_autosolve_allocation_history():
    """Allocation history should contain at least one round of decisions."""
    problem = _make_problem()
    result = autosolve(problem, StepSolution, timeout=3)

    history = result.logs["allocation_history"]
    assert len(history) >= 1
    # Each entry should map solver names to time slices.
    for entry in history:
        assert isinstance(entry, dict)
        for name, ts in entry.items():
            assert isinstance(name, str)
            assert ts > 0


def test_autosolve_independent_eval_counts():
    """Each solver in the default portfolio should have its own problem copy
    so eval_count tracking is independent."""
    problem = _make_problem()
    original_eval_count = problem.eval_count
    result = autosolve(problem, StepSolution, timeout=3)

    # After autosolve the original problem's eval_count should not have been
    # modified by the solvers (they each got dill.copy'd problems).
    assert problem.eval_count == original_eval_count


def test_autosolve_parallel_execution():
    """Verify solvers run in parallel via pathos ProcessPool.

    We use lightweight single-trajectory solvers to minimise serialisation
    overhead and a generous timeout so that at least 2 rounds complete
    reliably despite process-pool startup costs.
    """
    problem = _make_problem(n_dim=10)
    solvers = [
        RandomSearch(dill.copy(problem), StepSolution),
        HillClimber(dill.copy(problem), StepSolution, mutation_rate=0.2),
    ]
    timeout = 12.0
    result = autosolve(problem, StepSolution, timeout=timeout, algorithms=solvers)

    assert result.logs["rounds"] >= 2, (
        f"Expected >= 2 rounds in {timeout}s with parallel execution, "
        f"got {result.logs['rounds']}"
    )


def test_autosolve_solver_state_persists_across_rounds():
    """Solver state (best_solution, population) should carry over between
    rounds because the worker returns updated solvers to the main process."""
    problem = _make_problem(n_dim=10)
    solvers = [
        HillClimber(dill.copy(problem), StepSolution, mutation_rate=0.2),
    ]
    timeout = 12.0
    result = autosolve(problem, StepSolution, timeout=timeout, algorithms=solvers)

    # With multiple rounds, the solver should have improved.  The convergence
    # log should show monotonically non-decreasing scores.
    entries = list(result.logs["convergence"].values())[0]
    assert len(entries) >= 2, (
        f"Expected >= 2 rounds in {timeout}s, got {len(entries)}"
    )
    scores = [score for _, score in entries]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1], (
            f"Score decreased between rounds: {scores}"
        )


# ---- Differential Evolution tests ------------------------------------------

def test_de_basic():
    """DifferentialEvolution can be instantiated and iterated on StepProblem."""
    problem = _make_problem(n_dim=10)
    de = DifferentialEvolution(problem, StepSolution, popsize=6, F=0.8, CR=0.9)

    initial_best = de.best_solution.score
    for _ in range(20):
        de.step()

    # After 20 generations the best should be at least as good.
    assert de.best_solution.score >= initial_best


def test_de_population_size():
    """Population is maintained at exactly the requested size."""
    problem = _make_problem(n_dim=10)
    pop = 8
    de = DifferentialEvolution(problem, StepSolution, popsize=pop)

    assert len(de.population) == pop
    de.step()
    assert len(de.population) == pop


def test_de_no_genes_raises():
    """DifferentialEvolution raises TypeError for solutions without genes."""
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

    problem = _make_problem()
    NoGenesSolution.problem = problem
    with pytest.raises(TypeError, match="genes"):
        DifferentialEvolution(problem, NoGenesSolution, popsize=4)


def test_de_in_autosolve():
    """DifferentialEvolution should appear in the default portfolio."""
    problem = _make_problem()
    result = autosolve(problem, StepSolution, timeout=3)

    # At least one solver name should contain "DE".
    de_names = [
        name for name in result.logs["convergence"]
        if name.startswith("DE ")
    ]
    assert len(de_names) >= 1, (
        f"Expected DE in convergence log, got: "
        f"{list(result.logs['convergence'].keys())}"
    )


def test_de_genes_clipping():
    """Trial vectors should stay in [0, 1]."""
    problem = _make_problem(n_dim=5)
    de = DifferentialEvolution(problem, StepSolution, popsize=6, F=2.0, CR=1.0)

    # Run several steps with an aggressive F to exercise clipping.
    for _ in range(10):
        de.step()

    for sol in de.population:
        for g in sol.genes:
            assert 0.0 <= g <= 1.0, f"Gene {g} out of [0, 1]"


def test_autosolve_score_caching():
    """After autosolve, solution_cls.problem should point to the original
    problem, and best_solution.score should still be valid."""
    problem = _make_problem()
    original_problem = problem
    result = autosolve(problem, StepSolution, timeout=3)

    # solution_cls.problem should be restored to the caller's problem.
    assert StepSolution.problem is original_problem

    # The best solution should have a cached score that's valid.
    best = result.best_solution
    assert best._score is not None
    assert isinstance(best.score, (int, float))


# ---- TSP (no genes) -------------------------------------------------------

def test_autosolve_tsp():
    """autosolve works with TspSolution which has no genes attribute.

    DifferentialEvolution and PSO require solutions with a genes attribute;
    both should be gracefully skipped and the remaining portfolio should
    still produce a valid result.
    """
    problem = TspProblem.random_problem(n_dim=3, cities=8)
    result = autosolve(problem, TspSolution, timeout=3)

    assert isinstance(result, Result)
    assert result.best_solution is not None
    assert isinstance(result.best_solution.score, (int, float))
    assert result.logs["rounds"] >= 1

    # DE and PSO require genes; both should be skipped for TSP.
    de_names = [n for n in result.logs["convergence"] if n.startswith("DE ")]
    pso_names = [n for n in result.logs["convergence"] if n.startswith("PSO ")]
    assert len(de_names) == 0, f"DE should have been skipped for TSP but found: {de_names}"
    assert len(pso_names) == 0, f"PSO should have been skipped for TSP but found: {pso_names}"
