from blackopt.algorithms import HillClimber, GeneticAlgorithm
from blackopt.examples.problems import StepProblem, StepSolution
from blackopt.algorithms import RandomSearch

from blackopt.compare import compare_solvers, SolverFactory
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
        
        for sf, m_dict in zip(sfs, ms):
            assert isinstance(m_dict['score'], Metric)
            assert len(m_dict['score'].data) > 2