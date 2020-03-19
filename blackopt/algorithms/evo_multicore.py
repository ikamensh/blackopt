from typing import List, Tuple

from blackopt.abc.solver import Solver
from blackopt.algorithms import EvolutionaryAlgorithm

import pathos

import random


def iteration(inp: Tuple[EvolutionaryAlgorithm, int]):
    ga, steps = inp
    ga.solution_cls.problem = ga.problem
    ga.record = lambda : None
    ga.problem.eval_count = 0
    ga.solve(steps)
    ga.metrics = None
    return ga

class MulticoreGeneticAlgorithm(Solver):
    name = "Multicore_GA"

    def __init__(self, problem, solution_cls, *args, **kwargs):

        super().__init__(problem, solution_cls)
        self.args = args
        self.kwargs = kwargs
        self.pool = pathos.pools.ProcessPool()
        self.gas: List[EvolutionaryAlgorithm] = [EvolutionaryAlgorithm(problem, solution_cls, *args, **kwargs) for i in range(self.pool.ncpus)]

    def solve(self, steps):
        self.problem.eval_count = 0
        mapping_steps = int(steps ** (1 / 5))
        steps_per_pool = steps // (self.pool.ncpus * mapping_steps)

        print(steps_per_pool)
        print(mapping_steps)


        for i in range(mapping_steps):
            self.gas = self.pool.map(iteration, zip(self.gas, [steps_per_pool] * len(self.gas) ))

            if i > mapping_steps // 2:
                all_population = sum( (ga.population for ga in self.gas), [])
                random.shuffle(all_population)
                for i, ga in enumerate(self.gas):
                    ga.population = all_population[ga.popsize * i: ga.popsize*(i+1)]
                    ga._rank()

            self.best_solution = max(
                [s.best_solution for s in self.gas] + [self.best_solution], key=lambda x: x.score
            )
            self.problem.eval_count += steps_per_pool * self.pool.ncpus
            self.record()
            print(i * steps_per_pool * self.pool.ncpus)






