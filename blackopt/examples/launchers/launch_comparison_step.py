from blackopt.algorithms import HillClimber, EvolutionaryAlgorithm
from blackopt.examples.problems import StepProblem, StepSolution
from blackopt.algorithms import RandomSearch
from blackopt.util.document import generate_report

from blackopt.compare import compare_solvers, SolverFactory


problems = [StepProblem.random_problem(n) for n in [250, 1000, 3000]]

sfs = []

n_steps = 500
trials = 10

for problem in problems:
    sfs += [SolverFactory(EvolutionaryAlgorithm, problem, StepSolution, p, 1 / problem.n_dim, e) for p in [10, 30] for e in [0, 1]]
    sfs += [SolverFactory(HillClimber,problem, StepSolution, mutation_rate=2/problem.n_dim)]
    # sfs.append(SolverFactory(RandomSearch, problem, StepSolution))

    ms = compare_solvers(trials, n_steps, sfs)
    generate_report(problem, ms)
