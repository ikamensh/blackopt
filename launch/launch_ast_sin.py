from problems import ExpressionMatchProblem
from algorithms import RandomSearch
from algorithms import HillClimber
from algorithms import GeneticAlgorithm



expression = lambda x: x**3 + 2*x**2 - 15*x + 5

# import math
# expression = lambda x: math.sin(x)

problem = ExpressionMatchProblem(expression, -10, 10)
solver = GeneticAlgorithm(problem, 15, 0.5, 2)
# solver = HillClimber(problem, 0.5)
# solver = RandomSearch(problem)

progress_metric = solver.solve(10000)
answer : ExpressionMatchProblem.solution_cls  = solver.best_solution
answer.tree.write_down()


from ilya_ezplot import Metric, plot_group, ez_plot
ez_plot(progress_metric)

m = Metric(name="true")
m.add_arrays(problem.X, problem.Y)

y_pred = [answer.tree.eval(x) for x in problem.X]

m2 = Metric(name="predictions")
m2.add_arrays(problem.X, y_pred)

plot_group([m, m2], smoothen=False, name="true_vs_predicted")


# from cProfile import Profile
# profiler = Profile()
# profiler.runcall(solver.solve, 1000)
# profiler.print_stats('cumulative')


