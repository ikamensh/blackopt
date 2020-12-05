from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber, OffspringSelection, Rapga, Sasegasa
from blackopt.algorithms import EvolutionaryAlgorithm, SimAnneal
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(4e5)
n_trials = 6

cities = 60
problem = TspProblem.random_problem(2, cities)

solvers = []

solvers.append(
    SolverFactory(
        Rapga, problem, TspSolution, 300, 2 / cities, 0, 0.5
    )
)
solvers.append(
    SolverFactory(
        Rapga, problem, TspSolution, 300, 2 / cities, 0, 0.5, regularize=True
    )
)


if __name__ == "__main__":
    import time

    t = time.time()
    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
    print(time.time() - t)
