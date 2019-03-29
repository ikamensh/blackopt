from problems.problem import Problem, Solution
from algorithms.solver import Solver
from utils.plot import Metric
import numpy as np
from typing import List
import random

class GeneticAlgorithm(Solver):
    name = "GA"

    def __init__(self, problem: Problem, popsize: int, mutation_rate: float, elite_size: int,
                 heavy_tail_mutation = False):

        assert 0 < mutation_rate <= 1
        assert popsize > 1
        assert popsize > elite_size
        assert isinstance(elite_size, int)

        self.heavy_tail_mutation = heavy_tail_mutation

        self._mutation_rate = mutation_rate
        self.problem = problem
        self.population = [problem.random_solution() for _ in range(popsize)]
        self.popsize = popsize
        self.elite_size = elite_size

        self.generation = 1
        self.avg = None
        self.rank()


    @property
    def mutation_rate(self):
        if self.heavy_tail_mutation:
            lambd = 1 / self._mutation_rate
            return random.expovariate(lambd)
        else:
            return self._mutation_rate


    def solve(self, n_evaluations):

        best_score_metric = Metric(x_label="evaluations", y_label="best_score")

        while self.problem.eval_count < n_evaluations:


            next_generation = self.population[:self.elite_size]
            next_generation += self.breed(self.popsize - self.elite_size)
            self.population = next_generation

            self.rank()
            best_score_metric.add_record(
                self.problem.eval_count, self.best_solution.score)
            self.generation += 1

        print(f"{self} is Done in {self.generation} generations")
        return best_score_metric

    def rank(self):
        self.population = sorted(self.population, key=lambda x: x.score, reverse=True)
        self.best_solution: Solution = max(self.population, key=lambda x: x.score)
        self.avg = sum([x.score for x in self.population]) / len(self.population)

    def select_parents(self, n: int, smoothen_chances: float) -> List[Solution]:
        indexes = np.arange(0, len(self.population), dtype=np.int)
        chances = np.arange(len(self.population), 0, -1, dtype=np.int) + smoothen_chances / (1 - smoothen_chances + 1e-9)
        chances = chances / sum(chances)
        parent_indexes = np.random.choice(indexes, n, True, chances)
        parents = np.array(self.population)[parent_indexes]

        return parents

    def breed(self, n: int, smoothen_chances = 0) -> List[Solution]:

        parents = self.select_parents(n, smoothen_chances)
        children: List[Solution] = []

        for i in range(n):
            parent_1 = parents[i]
            parent_2 = parents[len(parents) - i - 1]
            children += parent_1.crossover(parent_2)

        children = [child.mutate(self.mutation_rate) for child in children]

        return children

    def __str__(self):
        return f"{self.name} with mut_rate - {self._mutation_rate} & " \
            f"pop_size - {self.popsize} & " \
            f"elite - {self.elite_size}" + (" HEAVY" if self.heavy_tail_mutation else "")