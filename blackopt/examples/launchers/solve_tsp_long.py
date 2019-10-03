from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import SasegasaContinuous

from blackopt.persistence import ContinuousOptimizer
from blackopt.compare import SolverFactory


# problem = TspProblem.random_problem(2, 130)
problem = TspProblem.load("Tsp 130 cities & 2 dim")

solver = SolverFactory(SasegasaContinuous, problem, TspSolution, 200, 1/130, 0)

co = ContinuousOptimizer(problem, solver)
co.run()
