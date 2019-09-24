from examples.problems import TspProblem, TspSolution
from blackbox.algorithms import RandomSearch
from blackbox.algorithms import GeneticAlgorithm
from blackbox.util.document import generate_report


from blackbox.compare import compare_solvers, SolverFactory

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
        for elite_size, color in zip({1, popsize // 10}, ["purple", "teal", "brown"]):
            for mr, linewidth in zip([5e-4, 5e-3, 0.01], [0.5, 1.1, 1.8]):
                style = {}
                # if heavy:
                #     style["dashes"] = [2, 2, 10, 2]
                style["linewidth"] = linewidth
                style["color"] = color
                ga = SolverFactory(
                    GeneticAlgorithm,
                    problem,
                    TspSolution,
                    popsize,
                    mr,
                    elite_size,
                    plot_kwargs=style,
                )
                solvers.append(ga)

    ms = compare_solvers(n_trials, n_steps, solvers)
    generate_report(problem, ms)
