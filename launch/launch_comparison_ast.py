from blackbox.algorithms import HillClimber
from util.document import PlotProgress
from blackbox.algorithms import RandomSearch
from blackbox.algorithms import GeneticAlgorithm
from problems import ExpressionMatchProblem
import pathos


# def solve_n_times(solver: Solver, n: int, n_steps: int):
#     metrics = []
#
#     for i in range(n):
#         solver.reset()
#         m = solver.solve(n_steps)
#         print(solver, i)
#         metrics.append(m)
#
#     return sum(metrics)

# n_steps = int(5e2)
# n_trials = 2

def maping(solver):
    metrics = []

    for i in range(10):
        solver.reset()
        m = solver.solve(int(5e4))
        print(solver, i)
        metrics.append(m)

    return sum(metrics)


if __name__ == "__main__":
    expression = lambda x: x ** 3 + 2 * x ** 2 - 15 * x + 5
    problem = ExpressionMatchProblem(expression, -10, 10)

    docu = PlotProgress(problem)

    solvers = [
        HillClimber(problem, mutation_rate=1),
        RandomSearch(problem),
        GeneticAlgorithm(problem, popsize=15, mutation_rate=0.5, elite_size=2),
    ]


    pool = pathos.pools.ProcessPool()

    metrics = pool.map(maping, solvers)
    docu.generate_report(metrics)
