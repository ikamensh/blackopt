from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, HillClimber
from blackopt.algorithms import GeneticAlgorithm, GeneticAlgorithmInject
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(2e6)
n_trials = 10

cities = 20
problem = TspProblem.random_problem(15, cities)

solvers = []
# solvers.append(SolverFactory(RandomSearch, problem, TspSolution))
solvers.append(SolverFactory(HillClimber, problem, TspSolution, 8 / cities))
solvers.append(SolverFactory(HillClimber, problem, TspSolution, 2 / cities))
solvers.append(SolverFactory(HillClimber, problem, TspSolution, 4 / cities))
# solvers.append(
#     SolverFactory(GeneticAlgorithmInject, problem, TspSolution, 50, 2 / cities, 1, 20)
# )
# solvers.append(
#     SolverFactory(GeneticAlgorithmInject, problem, TspSolution, 10, 2 / cities, 1, 20)
# )
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 3, 2 / cities, 1)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 50, 0.25 / cities, 0)
)
solvers.append(
    SolverFactory(GeneticAlgorithm, problem, TspSolution, 50, 0.125 / cities, 0)
)

if __name__ == "__main__":
    import time
    t = time.time()
    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
    print(time.time() - t)
