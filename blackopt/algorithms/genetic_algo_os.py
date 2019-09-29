from typing import List

from blackopt.abc import Solution
from blackopt.algorithms import GeneticAlgorithm


def keep(child_score: float, parent_min: float, diff: float, pressure: float):
    return child_score > parent_min + pressure * diff


class Gaos(GeneticAlgorithm):
    name = "Gaos"
    max_selective_pressure = 200

    def solve(self, steps):

        while self.problem.eval_count < steps:

            next_generation = self.population[: self.elite_size]
            next_generation += self._breed(
                self.popsize - self.elite_size,
                pressure=0.8 * self.problem.eval_count / steps,
            )
            self.population = next_generation

            self._rank()
            self.record()
            self.generation += 1
            if not self.generation % 10:
                print("Generation", self.generation, self.problem.eval_count)

        print(f"{self} is Done in {self.generation} generations")

    def record(self):
        super().record()
        self.record_metric("selective pressure", self.selective_pressure)


    def _breed(self, n: int, pressure=0.5) -> List[Solution]:

        result: List[Solution] = []
        ctr = 0
        while len(result) < n and ctr < self.max_selective_pressure:
            ctr += 1
            children = []
            parents = self._select_parents(n)

            parent_scores = {}  # child -> min_parent_score, parents_diff
            for i in range(n):
                parent_1 = parents[i]
                parent_2 = parents[len(parents) - i - 1]
                new = parent_1.crossover(parent_2)
                mutated = []
                for c in new:
                    c = c.mutate(self.mutation_rate)
                    mutated.append(c)
                    parent_scores[c] = (
                        min([parent_1.score, parent_2.score]),
                        abs(parent_1.score - parent_2.score),
                    )

                children += mutated

            result += [
                c for c in children if keep(c.score, *parent_scores[c], pressure)
            ]

        self.selective_pressure = ctr

        # if len(result) < n:
        #     result += [
        #         self.solution_cls.random_solution() for i in range(n - len(result))
        #     ]

        return result[:n]
