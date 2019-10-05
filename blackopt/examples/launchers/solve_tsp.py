from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import (
    RandomSearch,
    MulticoreRS,
    MulticoreGeneticAlgorithm,
    Sasegasa,
    Rapga,
    SasegasaContinuous,
)
from blackopt.util.document import generate_report


import time
from blackopt.config import prepend_rootdir
prepend_rootdir("~/PycharmProjects/optim")

problem = TspProblem.random_problem(2, 130)
problem.save()
# problem = TspProblem.load("Tsp 130 cities & 2 dim")


# solver = Rapga(problem, TspSolution, 100, 1 / 130)
# t = time.time()
# solver.solve(int(2e5))
# print("Duration: ", time.time() - t)
# generate_report(problem, {solver: solver.metrics})

# solver = SasegasaContinuous(problem, TspSolution, 200, 1 / 130)
# for i in range(10):
#     t = time.time()
#     solver.solve(int(2e5))
#     print(f"Step {i}, Duration: ", time.time() - t)
#
# generate_report(problem, {solver: solver.metrics})
