from blackbox.algorithms import HillClimber
from blackbox.problems import StepProblem, StepSolution
from blackbox.util.document import generate_report
from blackbox.algorithms import RandomSearch

problems = [StepProblem.random_problem(n) for n in [50, 250, 1000, 3000]]

for problem in problems:


    n_steps = int(2e4)

    metrics = []
    for mr in [0.01, 0.03, 0.1]:

        hc_metrics = []

        for i in range(5):
            solver = HillClimber(problem, StepSolution, mutation_rate=mr)
            solver.solve(n_steps)
            hc_metrics.append(solver.best_score_metric)
            print(solver, i)

        metrics.append(sum(hc_metrics))

    random_search_metrics = []

    for i in range(5):
        rs = RandomSearch(problem, StepSolution)
        rs.solve(n_steps)
        random_search_metrics.append(rs.best_score_metric)
        print(rs, i)

    metrics.append(sum(random_search_metrics))

    generate_report(problem, metrics)
