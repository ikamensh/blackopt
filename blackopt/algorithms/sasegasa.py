from blackopt.algorithms import Rapga
from blackopt.abc import Problem, Solver
from typing import List, Tuple


import pathos


def iteration(inp: Tuple[Rapga, int]):
    ga, steps = inp
    ga.solution_cls.problem = ga.problem
    ga.generation = 0
    ga.record = lambda : None
    ga.problem.eval_count = 0
    ga.solve(steps)
    ga.metrics = None
    return ga


class Sasegasa(Solver):
    name = "Sasegasa"
    def __init__(
        self,
        problem: Problem,
        solution_cls,
        popsize: int,
        mutation_rate: float,
        elite_size: int = 0,
        equal_chances: float = 0.5,
        growth_factor=30,
        n_villages=12,
    ):
        self.problem = problem
        self.solution_cls = solution_cls
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.equal_chances = equal_chances
        self.growth_factor = growth_factor
        self.n_villages = n_villages

        super().__init__(problem, solution_cls)

        self.pool = pathos.pools.ProcessPool()

    def solve(self, steps):
        self.problem.eval_count = 0
        runs = sum(range(self.n_villages + 1))

        steps_per_run = steps // runs
        print(f"Sasegasa will run with {steps_per_run} steps per epoch/village")

        villages = [
            Rapga(
                self.problem,
                self.solution_cls,
                self.popsize,
                self.mutation_rate,
                self.elite_size,
                self.equal_chances,
                self.growth_factor,
                early_stop=True
            ) for i in range(self.n_villages)
        ]

        while len(villages) > 1:
            villages: List[Rapga] = self.pool.map(iteration, zip(villages, [steps_per_run] * len(villages)))
            self.problem.eval_count += sum(v.problem.eval_count for v in villages)
            self.best_solution = max(
                [s.best_solution for s in villages] + [self.best_solution], key=lambda x: x.score
            )
            self.record()
            print(f"Sasegasa: {len(villages)}, {self.problem.eval_count}")

            population = []
            for v in villages:
                population += v.population

            self.n_villages -= 1
            new_size = len(population) // self.n_villages
            populations = [population[i * new_size: i * new_size + new_size] for i in range(self.n_villages)]
            homeless = population[new_size * self.n_villages:]
            populations[-1] += homeless

            villages = villages[:-1]
            for v, p in zip(villages, populations):
                v.population = p
                v.popsize = len(p)

        v = villages[0]
        iteration( (v, steps_per_run*3) )
        self.best_solution = max(
            [v.best_solution] + [self.best_solution], key=lambda x: x.score
        )
        self.problem.eval_count += v.problem.eval_count
        self.record()



