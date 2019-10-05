from typing import List

from blackopt.abc import Problem, Solution
from blackopt.algorithms import GeneticAlgorithm


class GeneticAlgorithmInject(GeneticAlgorithm):
    name = "GA"

    def __init__(
        self,
        problem: Problem,
        solution_cls,
        popsize: int,
        mutation_rate: float,
        elite_size: int,
        inject_quality: int,
    ):
        super().__init__(problem, solution_cls, popsize, mutation_rate, elite_size)
        self.inject_quality = inject_quality

    def _inject(self):
        best_solution = self.solution_cls.random_solution()
        for i in range(self.inject_quality):
            solution = self.solution_cls.random_solution()
            if solution.score > best_solution.score:
                best_solution = solution

        return best_solution

    def solve(self, steps):
        self.problem.eval_count = 0
        while self.problem.eval_count < steps:

            next_generation = self.population[: self.elite_size]
            next_generation += [self._inject()]
            next_generation += self._breed(self.popsize - self.elite_size - 1)
            self.population = next_generation

            self._rank()
            self.record()
            self.generation += 1
            if not self.generation % 100:
                print("Generation", self.generation, self.problem.eval_count)

        print(f"{self} is Done in {self.generation} generations")

    def _breed(self, n: int, smoothen_chances=0) -> List[Solution]:

        parents = self._select_parents(n, smoothen_chances)
        children: List[Solution] = []

        for i in range(n):
            parent_1 = parents[i]
            parent_2 = parents[len(parents) - i - 1]
            children += parent_1.crossover(parent_2)

        children = [child.mutate(self.mutation_rate) for child in children]

        return children

    def __str__(self):
        return (
            f"{self.name} with mut - {self.mutation_rate} & "
            f"pop - {self.popsize} & "
            f"elite - {self.elite_size}"
            f"inj - {self.inject_quality}"
        )
