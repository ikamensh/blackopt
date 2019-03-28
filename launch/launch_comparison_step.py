from algorithms import HillClimber
from problems import StepProblem
from algorithms.document import PlotProgress
from algorithms import RandomSearch

problems = [StepProblem.random_problem(n) for n in [50, 250, 1000, 3000]]

for problem in problems:

    best_score_metrics = {}

    docu = PlotProgress(problem)
    n_steps = int(2e4)

    for mr in [0.01, 0.03, 0.1]:

        metrics = []
        solver = HillClimber(problem, mutation_rate=mr)

        for i in range(5):
            solver.reset()
            m = solver.solve(n_steps)
            metrics.append(m)
            print(solver, i)

        best_score_metrics[str(solver)] = sum(metrics)


    random_search_metrics = []
    rs = RandomSearch(problem)
    for i in range(5):
        rs.reset()
        metric = rs.solve(n_steps)
        random_search_metrics.append(metric)
        print(solver, i)

    best_score_metrics[str(rs)] = sum(random_search_metrics)

    docu.generate_report(best_score_metrics)

