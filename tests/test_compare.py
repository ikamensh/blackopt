from blackbox.algorithms import HillClimber, GeneticAlgorithm
from blackbox.examples.problems import StepProblem, StepSolution
from blackbox.algorithms import RandomSearch

from blackbox.compare import compare_solvers, SolverFactory
from ilya_ezplot import Metric

def test_compare():
    problems = [StepProblem.random_problem(n) for n in [50, 250]]

    sfs = []

    n_steps = 50
    trials = 2

    for problem in problems:
        sfs += [SolverFactory(GeneticAlgorithm, problem, StepSolution, 10, 1/problem.n_dim, 1)]
        sfs += [SolverFactory(HillClimber,problem, StepSolution, mutation_rate=2/problem.n_dim)]
        sfs.append(SolverFactory(RandomSearch, problem, StepSolution))

        ms = compare_solvers(trials, n_steps, sfs)
        assert isinstance(ms[0], Metric)
        assert len(ms[0].data) > 2