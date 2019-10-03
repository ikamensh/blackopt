from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch, MulticoreRS, MulticoreGeneticAlgorithm, Sasegasa, Rapga
from blackopt.algorithms import GeneticAlgorithm
from blackopt.util.document import generate_report


import time

problem = TspProblem.random_problem(2, 90)

solver = Rapga(problem, TspSolution, 750, 1/90, 0)
t = time.time()
solver.solve(int(2e6))
print("Duration: ", time.time() - t)
generate_report(problem, {solver: solver.metrics})


# solver = GeneticAlgorithm(problem, TspSolution, 15, 1/200, 1)
# t = time.time()
# solver.solve(int(1e6))
# print("Duration: ", time.time() - t)
# generate_report(problem, {solver: solver.metrics})