from blackopt.algorithms import Gaos
from blackopt.abc import Problem
import random


def is_diverse(new, pop_sample, pressure):
    if len(pop_sample) == 0:
        return True
    avg_similarity = sum(p.similarity(new) for p in pop_sample) / len(pop_sample)
    return avg_similarity < 1 - pressure


class Rapga(Gaos):
    name = "Rapga"

    def __init__(
        self,
        problem: Problem,
        solution_cls,
        popsize: int,
        mutation_rate: float,
        elite_size: int,
        equal_chances: float = 0.5,
        growth_factor = 30
    ):
        super().__init__(problem, solution_cls, popsize, mutation_rate, elite_size, equal_chances)
        self.growth_factor = growth_factor



    def solve(self, steps):

        self.population += [
            self.solution_cls.random_solution()
            for i in range(self.growth_factor * self.popsize)
        ]
        self._rank()
        while self.problem.eval_count < steps:

            next_generation = self.population[: self.elite_size]

            for i in range(self.growth_factor):
                pressure = 0.8 * self.problem.eval_count / steps
                new = self._breed(self.popsize - self.elite_size, pressure=pressure)
                diversity_sample = (
                    next_generation
                    if len(next_generation) <= self.popsize * 3
                    else random.sample(next_generation, self.popsize * 3)
                )
                next_generation += [
                    c for c in new if is_diverse(c, diversity_sample, 0.15)
                ]
                if self.selective_pressure == self.max_selective_pressure:
                    break

            self.population = next_generation

            self._rank()
            self.record()
            self.generation += 1
            if not self.generation % 5:
                print("Generation", self.generation, self.problem.eval_count)

        print(f"{self} is Done in {self.generation} generations")


    def record(self):
        super().record()
        self.record_metric("actual popsize", self.actual_popsize)
