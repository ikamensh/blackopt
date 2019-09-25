from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber
from blackopt.algorithms import GeneticAlgorithm, GeneticAlgorithmInject
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(2e5)
n_trials = 5

cities = 200
problem = TspProblem.random_problem(2, cities)


solvers = []
solvers.append(SolverFactory(RandomSearch, problem, TspSolution))
solvers.append(SolverFactory(HillClimber, problem, TspSolution, 2 / cities))
solvers.append(
    SolverFactory(GeneticAlgorithmInject, problem, TspSolution, 50, 1 / cities, 1, 20)
)
solvers.append(
    SolverFactory(GeneticAlgorithmInject, problem, TspSolution, 10, 1 / cities, 1, 20)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 50, 1 / cities, 1)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 10, 1 / cities, 1)
)


ms = compare_solvers(n_trials, n_steps, solvers)
generate_report(problem, ms)
