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
        elite_size: int = 0,
        equal_chances: float = 0.5,
        max_selective_pressure: int = 200,
        early_stop=True,
        diversity_threshold=0.01,
        min_popsize=None,
    ):
        super().__init__(
            problem,
            solution_cls,
            popsize,
            mutation_rate,
            elite_size,
            equal_chances,
            max_selective_pressure,
            early_stop,
        )
        self.diversity_threshold = diversity_threshold
        self.min_popsize = min_popsize or ( popsize // 10 + 2)

    def check_early_stop(self):
        return super().check_early_stop() or self.actual_popsize < self.min_popsize

    def solve(self, steps):
        self.problem.eval_count = 0
        popsize_sqrt = int(self.popsize ** (1/2)) + 1
        self._rank()
        while self.problem.eval_count < steps and self.actual_popsize:

            next_generation = self.population[: self.elite_size]

            for i in range(popsize_sqrt):
                pressure = 0.8 * self.problem.eval_count / steps
                new = self._breed(popsize_sqrt, pressure=pressure)
                diversity_sample = (
                    next_generation
                    if len(next_generation) <= popsize_sqrt * 2
                    else random.sample(next_generation, popsize_sqrt * 2)
                )
                next_generation += [
                    c
                    for c in new
                    if is_diverse(c, diversity_sample, self.diversity_threshold)
                ]
                if self.selective_pressure == self.max_selective_pressure:
                    break

            self.population = next_generation

            self._rank()
            self.record()
            self.generation += 1
            if not self.generation % 5:
                print("Generation", self.generation, self.problem.eval_count)

            if self.early_stop and self.check_early_stop():
                break

        self.salut()

    def record(self):
        super().record()
        self.record_metric("actual popsize", self.actual_popsize)

    def __str__(self):
        return (
            f"{self.name} with mut_rate - {self.mutation_rate:.5f} & "
            f"pop_size - {self.popsize} & "
            f"elite - {self.elite_size} & equal_c - {self.equal_chances}"
            f"div threshold - {self.diversity_threshold}"

        )

