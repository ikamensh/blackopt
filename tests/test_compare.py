from blackopt.algorithms import HillClimber, EvolutionaryAlgorithm, SimAnneal, Gaos, Rapga
from blackopt.examples.problems.stepwise import StepProblem, StepSolution
from blackopt.algorithms import RandomSearch

from blackopt.compare import compare_solvers, SolverFactory
from ilya_ezplot import Metric

def test_compare():
    problems = [StepProblem.random_problem(n) for n in [50, 250]]

    sfs = []

    n_steps = 600
    trials = 2

    for problem in problems:
        sfs += [SolverFactory(EvolutionaryAlgorithm, problem, StepSolution, 3, 1 / problem.n_dim, 1)]
        sfs += [SolverFactory(Gaos, problem, StepSolution, 3, 1/problem.n_dim, 1)]
        sfs += [SolverFactory(Rapga, problem, StepSolution, 3, 1/problem.n_dim, 1)]
        sfs += [SolverFactory(HillClimber,problem, StepSolution, mutation_rate=2/problem.n_dim)]
        sfs += [SolverFactory(SimAnneal, problem, StepSolution, mutation_rate=2 / problem.n_dim)]
        sfs.append(SolverFactory(RandomSearch, problem, StepSolution))
        
        ms = compare_solvers(trials, n_steps, sfs)
        
        for sf, m_dict in ms.items():
            assert isinstance(m_dict['best_score'], Metric)
            assert len(m_dict['best_score'].data) > 2