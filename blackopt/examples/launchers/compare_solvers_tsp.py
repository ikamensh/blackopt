from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber, Gaos, Rapga
from blackopt.algorithms import GeneticAlgorithm, SimAnneal
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(3e5)
n_trials = 6

cities = 40
problem = TspProblem.random_problem(2, cities)

solvers = []
solvers.append(
    SolverFactory(Gaos, problem, TspSolution, 50, 2 / cities, 0)
)
solvers.append(
    SolverFactory(Gaos, problem, TspSolution, 10, 2 / cities, 0)
)
solvers.append(
    SolverFactory(Rapga, problem, TspSolution, 50, 2 / cities, 0)
)
solvers.append(
    SolverFactory(Rapga, problem, TspSolution, 50, 2 / cities, 1)
)
solvers.append(
    SolverFactory(Rapga, problem, TspSolution, 10, 2 / cities, 1)
)
solvers.append(
    SolverFactory(Rapga, problem, TspSolution, 10, 2 / cities, 0)
)



if __name__ == "__main__":
    import time
    t = time.time()
    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
    print(time.time() - t)
