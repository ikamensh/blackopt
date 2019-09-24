from __future__ import annotations
from blackopt.algorithms import HillClimber
from blackopt.examples.problems import BumpyProblem, BumpySolution
from blackopt.algorithms import RandomSearch
from blackopt.compare import compare_solvers, SolverFactory
from blackopt.util.document import generate_report


problem = BumpyProblem.random_problem(100, 200)


solvers = [
    SolverFactory(HillClimber, problem, BumpySolution, mutation_rate=mr) for mr in [0.01, 0.03, 0.1]
]
solvers.append(SolverFactory(RandomSearch, problem, BumpySolution))

n_steps = int(1e4)
n_trials = 12

import time
t = time.time()
ms = compare_solvers(n_trials, n_steps, solvers)
print(time.time() - t)

generate_report(problem, ms)


