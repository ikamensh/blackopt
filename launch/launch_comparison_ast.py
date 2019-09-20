from algorithms import HillClimber
from algorithms.document import PlotProgress
from algorithms import RandomSearch
from algorithms import GeneticAlgorithm
from algorithms.solver import Solver
from problems import ExpressionMatchProblem
import pathos


def solve_n_times(solver: Solver, n: int, n_steps: int):
    metrics = []

    for i in range(n):
        solver.reset()
        m = solver.solve(n_steps)
        print(solver, i)
        metrics.append(m)

    return sum(metrics)


def maping(solver):
    return solve_n_times(solver, n=n_trials, n_steps=n_steps)


if __name__ == "__main__":
    expression = lambda x: x ** 3 + 2 * x ** 2 - 15 * x + 5
    problem = ExpressionMatchProblem(expression, -10, 10)

    docu = PlotProgress(problem)

    solvers = [
        HillClimber(problem, mutation_rate=1),
        RandomSearch(problem),
        GeneticAlgorithm(problem, popsize=15, mutation_rate=0.5, elite_size=2),
    ]

    n_steps = int(2e3)
    n_trials = 10

    pool = pathos.pools.ProcessPool()

    metrics = pool.map(maping, solvers)
    docu.generate_report(metrics)
