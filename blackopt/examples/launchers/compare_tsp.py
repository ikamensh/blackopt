from blackopt.examples.problems import TspProblem, TspSolution
from blackopt.algorithms import RandomSearch
from blackopt.algorithms import EvolutionaryAlgorithm
from blackopt.util.document import generate_report


from blackopt.compare import compare_solvers, SolverFactory

n_steps = int(2e3)
n_trials = 3

problems = [
    TspProblem.random_problem(n_dim, cities)
    for n_dim in [2, 10, 50]
    for cities in [40, 200, 1000]
]

for problem in problems:

    solvers = []
    solvers.append(SolverFactory(RandomSearch, problem, TspSolution))
    for popsize in [10, 50]:
        for elite_size in [1, popsize // 10]:
            for mr in [5e-4, 5e-3, 0.01]:
                # if heavy:
                #     style["dashes"] = [2, 2, 10, 2]
                # style["linewidth"] = linewidth
                # style["color"] = color
                ga = SolverFactory(
                    EvolutionaryAlgorithm,
                    problem,
                    TspSolution,
                    popsize,
                    mr,
                    elite_size,
                )
                solvers.append(ga)

    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
