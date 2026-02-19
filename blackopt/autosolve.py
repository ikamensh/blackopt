"""Automatic solver portfolio: runs multiple algorithms with adaptive time
allocation and returns the overall best solution found within a wall-clock
budget.

Usage::

    from blackopt.autosolve import autosolve
    result = autosolve(problem, SolutionClass, timeout=30)
    print(result.best_solution.score)
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import dill
import pathos.pools

from blackopt.abc import Problem, Solution, Solver
from blackopt.exceptions import EarlyStopException


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Returned by :func:`autosolve`."""

    best_solution: Solution
    logs: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Worker function – executed inside the process pool
# ---------------------------------------------------------------------------

# Minimum wall-clock seconds a solver receives per round.  Even the heaviest
# population-based step (crossover + mutation + ranking on a moderate
# population) completes well within 0.1 s, so this floor prevents starvation
# without wasting time on solvers that have already converged.
_MIN_TIME_SLICE: float = 0.1


def _run_solver_round(args: Tuple[Solver, float]) -> Tuple[Solver, float, int]:
    """Run *solver* for up to *time_budget* seconds inside a worker process.

    Returns ``(solver, best_score, steps_completed)`` so the main process can
    update its bookkeeping and maintain solver state across rounds.

    ``Solution.problem`` is re-bound at the start of every worker invocation
    to guarantee correctness after a ``fork``/``pickle`` boundary.
    """
    solver, time_budget = args

    # Re-bind the class variable that is lost across the process boundary.
    solver.solution_cls.problem = solver.problem

    steps = 0
    deadline = time.monotonic() + time_budget
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            while time.monotonic() < deadline:
                solver.step()
                steps += 1
        except EarlyStopException:
            pass

    # Cache all scores *before* returning so they survive re-binding of
    # ``Solution.problem`` when the next solver runs in the same worker
    # process.  Accessing ``.score`` triggers lazy evaluation and stores
    # the value in ``_score``.
    for sol in solver.population:
        _ = sol.score
    _ = solver.best_solution.score

    return solver, solver.best_solution.score, steps


# ---------------------------------------------------------------------------
# Default portfolio builder
# ---------------------------------------------------------------------------

def _default_portfolio(
    problem: Problem,
    solution_cls: Type[Solution],
) -> List[Solver]:
    """Instantiate one solver of every ``step()``-compatible type with
    sensible defaults derived from *problem*.

    After all solvers are created, every initial ``best_solution`` (and
    population member) has its score force-evaluated so that it is cached
    in ``_score`` and will not need lazy re-evaluation later – avoiding
    contamination from the ``solution_cls.problem`` class variable being
    overwritten by successive solver constructors.

    Excluded algorithms
    -------------------
    * **Sasegasa / SasegasaContinuous** – These solvers manage multiple
      parallel sub-populations ("villages") that progressively merge.  Their
      architecture requires the full ``solve(steps)`` lifecycle and cannot
      be decomposed into single ``step()`` calls, so they raise
      ``NotImplementedError`` from ``step()``.  Including them would require
      a fundamentally different scheduling strategy (whole-``solve()``
      delegation with a time limit), which is outside the scope of the
      round-robin ``step()``-based scheduler used here.
    * **MulticoreRS / MulticoreGeneticAlgorithm** – These override
      ``solve()`` entirely with their own multiprocessing and do not
      expose a meaningful ``step()``.
    """

    from blackopt.algorithms import (
        DifferentialEvolution,
        EvolutionaryAlgorithm,
        HillClimber,
        OffspringSelection,
        ParticleSwarmOptimization,
        RandomSearch,
        Rapga,
        SimAnneal,
        TabuSearch,
        TruncationSelectionGA,
        VariableNeighborhoodSearch,
    )

    # Heuristic mutation rate – roughly 1 gene flip per individual.
    n_dim = getattr(problem, "n_dim", 10)
    mr = min(1.0, 2.0 / max(n_dim, 1))

    solvers: List[Solver] = []

    # Each solver receives its own *deep copy* of the problem so that
    # ``eval_count`` tracking is independent across workers.
    #
    # IMPORTANT: every ``Solver.__init__`` sets
    # ``solution_cls.problem = problem`` (its own copy), so the class
    # variable is overwritten with each new solver.  We force-cache all
    # scores at the end of this function to make earlier solvers immune
    # to this overwrite.

    # --- single-trajectory ---------------------------------------------------
    solvers.append(RandomSearch(dill.copy(problem), solution_cls))
    solvers.append(HillClimber(dill.copy(problem), solution_cls, mutation_rate=mr))
    solvers.append(SimAnneal(dill.copy(problem), solution_cls, mutation_rate=mr))
    solvers.append(TabuSearch(dill.copy(problem), solution_cls, mutation_rate=mr))
    solvers.append(VariableNeighborhoodSearch(dill.copy(problem), solution_cls))

    # --- population-based ----------------------------------------------------
    pop = max(4, min(n_dim, 50))
    elite = max(1, pop // 10)
    solvers.append(
        EvolutionaryAlgorithm(
            dill.copy(problem), solution_cls,
            popsize=pop, mutation_rate=mr, elite_size=elite,
        )
    )
    solvers.append(
        OffspringSelection(
            dill.copy(problem), solution_cls,
            popsize=pop, mutation_rate=mr, elite_size=elite,
        )
    )
    solvers.append(
        Rapga(
            dill.copy(problem), solution_cls,
            popsize=pop, mutation_rate=mr, elite_size=elite,
        )
    )
    top_k = max(2, pop // 3)
    solvers.append(
        TruncationSelectionGA(
            dill.copy(problem), solution_cls,
            popsize=pop, top_k=top_k, mutation_rate=mr,
        )
    )
    # DifferentialEvolution and PSO require solutions with a ``genes``
    # attribute.  Gracefully skip them for solution classes that don't
    # support this (e.g. TspSolution which uses ``route`` instead).
    try:
        solvers.append(
            DifferentialEvolution(
                dill.copy(problem), solution_cls,
                popsize=pop, F=0.8, CR=0.9,
            )
        )
    except TypeError:
        pass
    try:
        solvers.append(
            ParticleSwarmOptimization(
                dill.copy(problem), solution_cls,
                popsize=pop, w=0.7, c1=1.5, c2=1.5,
            )
        )
    except TypeError:
        pass

    # Force-cache scores on every solution so they survive future
    # re-bindings of ``solution_cls.problem``.
    for solver in solvers:
        _ = solver.best_solution.score
        for sol in solver.population:
            _ = sol.score

    return solvers


# ---------------------------------------------------------------------------
# Core scheduler
# ---------------------------------------------------------------------------

def autosolve(
    problem: Problem,
    solution_cls: Type[Solution],
    timeout: float = 30.0,
    algorithms: Optional[List[Solver]] = None,
    patience: int = 5,
) -> Result:
    """Run a portfolio of solvers with adaptive time-slice scheduling.

    Parameters
    ----------
    problem : Problem
        The optimisation problem to solve.
    solution_cls : type
        Concrete :class:`Solution` subclass whose ``random_solution``,
        ``mutate``, and ``crossover`` methods are implemented.
    timeout : float
        Wall-clock budget in seconds (default 30).
    patience : int
        Stop early if the global best score has not improved for this many
        rounds (default 5). Set to 0 or a large value to disable.
    algorithms : list[Solver] | None
        Pre-built solver instances to use.  When *None* a default portfolio
        of all step()-compatible solvers is created automatically.  If you
        supply your own list, give each solver its own ``dill.copy(problem)``
        so evaluation counters stay independent.

    Returns
    -------
    Result
        Contains ``best_solution`` and ``logs`` with convergence data per
        algorithm and the resource allocation history.
    """
    if algorithms is not None:
        solvers = list(algorithms)
        # Force-cache scores so they are immune to solution_cls.problem
        # re-binding across solvers that share the same Solution subclass.
        for solver in solvers:
            _ = solver.best_solution.score
            for sol in solver.population:
                _ = sol.score
    else:
        solvers = _default_portfolio(problem, solution_cls)

    n = len(solvers)
    if n == 0:
        raise ValueError("At least one solver is required.")

    # -- bookkeeping ----------------------------------------------------------
    # Capture solver names *once* so that solvers whose __str__ includes
    # mutable state (e.g. Rapga shows actual_popsize) stay stable.
    # Append an index to guarantee uniqueness – two solvers with the same
    # __str__ (e.g. duplicate algorithms in a user-supplied portfolio) would
    # otherwise collide as convergence-log dict keys.
    solver_names: List[str] = [f"{s} #{i}" for i, s in enumerate(solvers)]

    # Initial time slice: 1 s or the fair share of the total budget,
    # whichever is smaller – but never less than _MIN_TIME_SLICE.
    initial_slice = max(_MIN_TIME_SLICE, min(1.0, timeout / n))
    time_slices: List[float] = [initial_slice] * n

    # Solvers that have been permanently retired (EarlyStop, etc.).
    retired: List[bool] = [False] * n

    # Previous-round best score per solver (to detect improvement).
    prev_scores: List[Optional[float]] = [None] * n

    # Per-algorithm convergence log: list of (wall_time, best_score) tuples.
    convergence: Dict[str, List[Tuple[float, float]]] = {
        name: [] for name in solver_names
    }
    # History of time allocation decisions.
    allocation_history: List[Dict[str, float]] = []
    # Round leaders: (round_index, leader_name, leader_score,
    #                 most_improved_name, most_improved_delta) per round.
    winners: List[Tuple[int, str, float, str, float]] = []

    global_best_solution: Solution = max(
        (s.best_solution for s in solvers), key=lambda s: s.score,
    )

    # Cap for adaptive time-slice growth.  No single solver may receive
    # more than half the total timeout in a single round – this prevents
    # a persistently-improving solver from monopolising the budget and
    # starving the rest of the portfolio.
    max_time_slice = timeout / 2.0

    start = time.monotonic()

    # -- main scheduling loop -------------------------------------------------
    # The pool is created inside the ``try`` block so that any exception
    # during construction is still safe (``pool`` will be ``None`` and the
    # ``finally`` block handles that).  We size the pool to the number of
    # solvers so each one gets its own worker process, avoiding cross-call
    # state pollution from the pathos singleton pool.
    pool: Optional[pathos.pools.ProcessPool] = None
    round_num = 0
    prev_global_best: float = global_best_solution.score
    rounds_without_improvement = 0
    try:
        pool = pathos.pools.ProcessPool(nodes=n)

        while True:
            elapsed = time.monotonic() - start
            remaining = timeout - elapsed
            if remaining <= 0:
                break

            # Determine which solvers will participate this round.
            active_indices = [i for i in range(n) if not retired[i]]
            if not active_indices:
                break

            # Compute dispatch slices for this round.  We must NOT mutate
            # ``time_slices`` here – those are the *base* values maintained by
            # the adaptive reallocation logic below.  Scaling in-place would
            # permanently shrink them, making the adaptive 2×/0.5× adjustments
            # start from an already-reduced baseline and starving solvers of
            # time in subsequent rounds.
            active_slices = {i: time_slices[i] for i in active_indices}
            total_requested = sum(active_slices.values())
            if total_requested > remaining:
                scale = remaining / total_requested
                active_slices = {i: ts * scale for i, ts in active_slices.items()}

            # Build work items for the pool.
            round_alloc: Dict[str, float] = {}
            work: List[Tuple[Solver, float]] = []
            work_indices: List[int] = []

            for i in active_indices:
                ts = active_slices[i]
                if ts < 0.001:
                    continue
                round_alloc[solver_names[i]] = ts
                work.append((solvers[i], ts))
                work_indices.append(i)

            if not work:
                break

            allocation_history.append(round_alloc)

            # Run all solvers in parallel.
            results = pool.map(_run_solver_round, work)

            wall_after = time.monotonic() - start

            # Collect results back into main-process state.
            for idx_in_work, (updated_solver, score, steps_completed) in enumerate(results):
                i = work_indices[idx_in_work]
                solvers[i] = updated_solver
                convergence[solver_names[i]].append((wall_after, score))

                if score > global_best_solution.score:
                    global_best_solution = updated_solver.best_solution

                # Retire solvers that completed zero steps (EarlyStop fired
                # before any work was done, meaning they are exhausted).
                if steps_completed == 0:
                    retired[i] = True

            # -- round leader and most improved ------------------------------
            round_leader_idx = max(
                work_indices,
                key=lambda i: solvers[i].best_solution.score,
            )
            leader_name = solver_names[round_leader_idx]
            leader_score = solvers[round_leader_idx].best_solution.score

            # Most improved: solver that contributed to global best, or
            # the one with highest (cur - prev) delta.
            global_improved = (
                prev_global_best is not None
                and global_best_solution.score > prev_global_best
            )
            if global_improved:
                contrib_idx = next(
                    i for i in work_indices
                    if solvers[i].best_solution.score == global_best_solution.score
                )
                most_improved_name = solver_names[contrib_idx]
                most_improved_delta = (
                    global_best_solution.score - prev_global_best
                )
            else:
                def _prev_baseline(i: int) -> float:
                    prev = prev_scores[i]
                    if prev is not None:
                        return prev
                    return solvers[i].best_solution.score

                best_delta_idx = max(
                    work_indices,
                    key=lambda i: (
                        solvers[i].best_solution.score - _prev_baseline(i)
                    ),
                )
                most_improved_name = solver_names[best_delta_idx]
                most_improved_delta = (
                    solvers[best_delta_idx].best_solution.score
                    - _prev_baseline(best_delta_idx)
                )

            winners.append((
                round_num,
                leader_name,
                leader_score,
                most_improved_name,
                most_improved_delta,
            ))

            prev_global_best = global_best_solution.score
            if global_improved:
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            # -- early stop on patience --------------------------------------
            if patience > 0 and rounds_without_improvement >= patience:
                break

            # -- adaptive reallocation ----------------------------------------
            # Only adjust solvers that actually ran this round.  Solvers
            # skipped due to tiny time slices (< 0.001) should not be
            # penalised – their prev_scores stay unchanged so they get a
            # fair comparison in the next round they participate in.
            for i in work_indices:
                if retired[i]:
                    continue
                cur_score = solvers[i].best_solution.score
                if prev_scores[i] is not None:
                    improved = cur_score > prev_scores[i]
                    time_slices[i] *= 2.0 if improved else 0.5
                    time_slices[i] = max(
                        _MIN_TIME_SLICE,
                        min(time_slices[i], max_time_slice),
                    )
                prev_scores[i] = cur_score

            round_num += 1

            if time.monotonic() - start >= timeout:
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
            pool.clear()

    # -- assemble result ------------------------------------------------------
    # Re-bind solution_cls.problem in the main process so the returned
    # solution's score property works correctly for the caller.
    solution_cls.problem = problem

    logs = {
        "convergence": convergence,
        "allocation_history": allocation_history,
        "winners": winners,
        "rounds": round_num,
    }

    return Result(best_solution=global_best_solution, logs=logs)
