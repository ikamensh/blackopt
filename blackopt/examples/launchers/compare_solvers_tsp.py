from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber, Gaos, Rapga, Sasegasa
from blackopt.algorithms import GeneticAlgorithm, SimAnneal
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(1e5)
n_trials = 3

cities = 40
problem = TspProblem.random_problem(2, cities)

solvers = []

solvers.append(
    SolverFactory(
        Rapga, problem, TspSolution, 600, 2 / cities, 0, 0.5
    )
)
solvers.append(
    SolverFactory(
        Sasegasa, problem, TspSolution, 150, 2 / cities, 0, 0.5, n_villages=12
    )
)


if __name__ == "__main__":
    import time

    t = time.time()
    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
    print(time.time() - t)
