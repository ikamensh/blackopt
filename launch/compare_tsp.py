from algorithms import HillClimber
from problems import TspProblem
from algorithms.document import PlotProgress
from algorithms import RandomSearch
from algorithms import GeneticAlgorithm
from algorithms.solver import Solver
from concurrent import futures


def solve_n_times(solver: Solver, n: int, n_steps: int):
    metrics = []

    for i in range(n):
        solver.reset()
        m = solver.solve(n_steps)
        print(solver, i)
        metrics.append(m)

    return sum(metrics)


problem = TspProblem.random_problem(n_dim=5, cities=200)
docu = PlotProgress(problem)

solvers = []
for popsize in [10, 50, 250]:
    # solvers += [HillClimber(problem, mutation_rate=mr) for mr in [0.002, 0.005, 0.01]]
    solvers.append(RandomSearch(problem))
    solvers += [GeneticAlgorithm(problem, popsize, mr, elite_size, heavy_tail_mutation=heavy)
                for mr in [5e-3, 0.01, 0.03]
                for elite_size in [0, 1, popsize//10]
                for heavy in [True, False]]


    best_score_metrics = {}

    n_steps = int(5e2)
    n_trials = 2

    pool = futures.ProcessPoolExecutor()


    def maping(solver):
        try:
            return solve_n_times(solver, n=n_trials, n_steps=n_steps)
        except:
            return None


    metrics = pool.map(maping, solvers)

    for solver, result in zip(solvers, metrics):
        if result:
            best_score_metrics[str(solver)] = result

    # for solver in solvers:
    #     best_score_metrics[str(solver)] = solve_n_times(solver, n = n_trials, n_steps=n_steps)

    docu.generate_report(best_score_metrics)
