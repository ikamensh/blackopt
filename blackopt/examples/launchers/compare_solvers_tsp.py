from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber
from blackopt.algorithms import GeneticAlgorithm, SimAnneal
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(5e4)
n_trials = 16

cities = 40
problem = TspProblem.random_problem(2, cities)

solvers = []
solvers.append(SolverFactory(SimAnneal, problem, TspSolution, 3 / cities))
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 3, 2 / cities, 1)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 3, 3 / cities, 1)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 2, 3 / cities, 1)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 2, 2 / cities, 1)
)


if __name__ == "__main__":
    import time
    t = time.time()
    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
    print(time.time() - t)
