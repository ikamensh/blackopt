from blackopt.algorithms import Rapga
from blackopt.abc import Problem, Solver
from typing import List, Tuple

import random

import pathos


def iteration(inp: Tuple[Rapga, int]):
    ga, steps = inp
    ga.solution_cls.problem = ga.problem
    ga.generation = 0
    ga.record = lambda: None
    ga.problem.eval_count = 0
    ga.solve(steps)
    ga.metrics = None
    return ga


class SasegasaContinuous(Solver):
    name = "SasegasaContinuous"

    def __init__(
        self,
        problem: Problem,
        solution_cls,
        popsize: int,
        mutation_rate: float,
        elite_size: int = 0,
        equal_chances: float = 0.5,
        max_selective_pressure: int = 200,
        diversity_threshold = 0.01,
        n_villages = None


    ):
        self.problem = problem
        self.solution_cls = solution_cls
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.equal_chances = equal_chances
        self.max_selective_pressure = max_selective_pressure
        self.pool = pathos.pools.ProcessPool()
        self.n_villages = n_villages or int(self.pool.ncpus * 1.2)

        self.villages = [
            Rapga(
                self.problem,
                self.solution_cls,
                self.popsize,
                self.mutation_rate,
                self.elite_size,
                self.equal_chances,
                diversity_threshold=diversity_threshold,
                max_selective_pressure = max_selective_pressure,
                early_stop=True,
            )
            for i in range(self.n_villages)
        ]

        super().__init__(problem, solution_cls)


    @property
    def population(self):
        population = []
        for v in self.villages:
            random.shuffle(v.population)
            population += v.population
        return population


    def solve(self, steps):

        steps_per_run = steps // self.n_villages
        print(f"Sasegasa will run with {steps_per_run} steps per epoch/village")


        self.villages: List[Rapga] = self.pool.map(
            iteration, zip(self.villages, [steps_per_run] * len(self.villages))
        )
        self.best_solution = max(
            [s.best_solution for s in self.villages] + [self.best_solution],
            key=lambda x: x.score,
        )

        self.problem.eval_count += sum(v.problem.eval_count for v in self.villages)
        self.record()

        population = self.population
        shift = random.randint(1, len(population))
        population = population[shift:] + population[:shift]
        populations = [
            population[i * self.popsize : i * self.popsize + self.popsize]
            for i in range(self.n_villages)
        ]
        homeless = population[self.popsize * self.n_villages :]
        populations[-1] += homeless

        for v, p in zip(self.villages, populations):
            v.population = p
