from blackopt.algorithms import OffspringSelection
from blackopt.abc import Problem
from blackopt.exceptions import EarlyStopException
import random


def average_similarity(new, pop_sample):
    if len(pop_sample) == 0:
        return 0
    return sum(p.similarity(new) for p in pop_sample) / len(pop_sample)

def is_diverse(avg_similarity, pressure):
    return avg_similarity < 1 - pressure

default_elite = 0
default_eq_ch = 0.5
default_max_sel_pressure = 200
default_div_threshold = 0.01


class Rapga(OffspringSelection):
    name = "Rapga"

    def __init__(
        self,
        problem: Problem,
        solution_cls,
        popsize: int,
        mutation_rate: float,
        elite_size: int = default_elite,
        equal_chances: float = default_eq_ch,
        max_selective_pressure: int = default_max_sel_pressure,
        early_stop=True,
        diversity_threshold=default_div_threshold,
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
        while self.problem.eval_count < steps and self.actual_popsize:
            pressure = 0.8 * self.problem.eval_count / steps
            self.step(pressure)

        self.salut()


    def step(self, pressure = 0.5):
        popsize_sqrt = int(self.popsize ** (1 / 2)) + 1
        next_generation = self.population[: self.elite_size]

        while len(next_generation) < self.popsize:

            new = self._breed(popsize_sqrt, pressure=pressure)
            diversity_sample = (
                next_generation
                if len(next_generation) <= popsize_sqrt * 2
                else random.sample(next_generation, popsize_sqrt * 2)
            )
            similarities = {}
            for n in new:
                similarities[n] = average_similarity(n, diversity_sample)
                n.regularization_score -= 0.2 * self.problem.score_span * similarities[n]

            next_generation += [
                c
                for c in new
                if is_diverse(similarities[c], self.diversity_threshold)
            ]
            if self.selective_pressure >= self.max_selective_pressure:
                raise EarlyStopException()

        self.population = next_generation
        self._rank()
        self.record()
        self.generation += 1
        if not self.generation % 5:
            print("Generation", self.generation, self.problem.eval_count)

        if self.early_stop and self.check_early_stop():
            raise EarlyStopException()

    def record(self):
        super().record()
        self.record_metric("actual popsize", self.actual_popsize)

    def __str__(self):
        string = f"{self.name} with mut_rate - {self.mutation_rate:.5f} & " \
            f"pop_size - {self.actual_popsize} "
        if self.elite_size != default_elite:
            string += f" & elite - {self.elite_size}"
        if self.equal_chances != default_eq_ch:
            string += f" & equal_c - {self.equal_chances}"
        if self.diversity_threshold != default_div_threshold:
            f" & div threshold - {self.diversity_threshold}"

        return string



