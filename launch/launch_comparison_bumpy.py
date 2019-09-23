from blackbox.algorithms import HillClimber
from problems import BumpyProblem
from blackbox.util.document import PlotProgress
from blackbox.algorithms import RandomSearch
from blackbox.abc.solver import Solver
from concurrent import futures


def solve_n_times(solver: Solver, n: int, n_steps: int):
    metrics = []

    for i in range(n):
        solver.reset()
        m = solver.solve(n_steps)
        print(solver, i)
        metrics.append(m)

    return sum(metrics)


problem = BumpyProblem.random_problem(100, 200)
docu = PlotProgress(problem)


solvers = [HillClimber(problem, mutation_rate=mr) for mr in [0.01, 0.03, 0.1]]
solvers.append(RandomSearch(problem))


best_score_metrics = {}

n_steps = int(2e4)
n_trials = 3

pool = futures.ProcessPoolExecutor()


def maping(solver):
    return solve_n_times(solver, n=n_trials, n_steps=n_steps)


metrics = pool.map(maping, solvers)

for solver, result in zip(solvers, metrics):
    best_score_metrics[str(solver)] = result

# for solver in solvers:
#     best_score_metrics[str(solver)] = solve_n_times(solver, n = n_trials, n_steps=n_steps)

docu.generate_report(best_score_metrics)
