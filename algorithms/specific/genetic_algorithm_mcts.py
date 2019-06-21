from problems.problem import Problem, Solution
from algorithms.solver import Solver
from ilya_ezplot import Metric
import numpy as np
from typing import List, Dict

from ControlledParam import ControlledParam


def _half_popsize(ga):
    return max(2, ga.popsize // 2)


class GeneticAlgorithmMcts(Solver):
    name = "GA MCTS"

    mutation_rate = ControlledParam(param_name='mutation_rate',
                                    min=-10, max=-1, exponential=True)

    smoothen_chances = ControlledParam(param_name='smoothen_chances',
                                    min=-1, max=6, exponential=True)

    elite_size = ControlledParam(param_name="elite_size",
                                 min=1, max=_half_popsize, integer=True)

    ctrl_params = [mutation_rate, smoothen_chances, elite_size]

    def __init__(self, problem: Problem, popsize: int, plot_kwargs: Dict = None):

        assert popsize > 1

        self.problem = problem
        self.popsize = popsize
        self.population = [problem.random_solution() for _ in range(popsize)]

        self.generation = 1
        self.avg = None
        self.plot_kwargs = plot_kwargs or {}
        self.rank()
        ControlledParam.init(self, self.ctrl_params)


    def solve(self, n_evaluations):

        best_score_metric = Metric(x_label="evaluations", y_label="best_score",
                                   style_kwargs=self.plot_kwargs)
        best_score_metric.add_record(
            self.problem.eval_count, self.best_solution.score)

        score_before = self.best_solution.score

        while self.problem.eval_count < n_evaluations:


            next_generation = self.population[:self.elite_size]
            next_generation += self.breed(self.popsize - self.elite_size)
            self.population = next_generation

            self.rank()
            best_score_metric.add_record(
                self.problem.eval_count, self.best_solution.score)
            self.generation += 1

            if self.generation % 10 == 0:
                delta = self.best_solution.score - score_before
                ControlledParam.step(self, delta)
                score_before = self.best_solution.score

        print(f"{self} is Done in {self.generation} generations")
        return best_score_metric

    def rank(self):
        self.population = sorted(self.population, key=lambda x: x.score, reverse=True)
        self.best_solution: Solution = max(self.population, key=lambda x: x.score)

    def select_parents(self, n: int) -> List[Solution]:
        indexes = np.arange(0, len(self.population), dtype=np.int)
        chances = np.arange(len(self.population), 0, -1, dtype=np.int) + self.smoothen_chances
        chances = chances / sum(chances)
        parent_indexes = np.random.choice(indexes, n, True, chances)
        parents = np.array(self.population)[parent_indexes]

        return parents

    def breed(self, n: int) -> List[Solution]:

        parents = self.select_parents(n)
        children: List[Solution] = []

        for i in range(n):
            parent_1 = parents[i]
            parent_2 = parents[len(parents) - i - 1]
            children += parent_1.crossover(parent_2)

        children = [child.mutate(self.mutation_rate) for child in children]

        return children

    def reset(self):
        self.population = [self.problem.random_solution() for _ in range(self.popsize)]
        self.generation = 1
        self.problem.eval_count = 0
        self.rank()
        ControlledParam.init(self, self.ctrl_params)
        # self.avg = None

    def __str__(self):
        return f"{self.name} with pop_size {self.popsize}"
