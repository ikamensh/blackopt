from __future__ import annotations
from typing import List
import random
from functools import lru_cache

from problems.problem import Problem, Solution

from problems.tsp.city import City
import math


class TspProblem(Problem):
    name = "Tsp"

    def __init__(self, cities : List[City]):
        self.cities = cities
        self.n_dim = len( cities[0].coordinates )
        self.eval_count = 0
        self.max_dist = len(self.cities) * math.sqrt(self.n_dim)

    @staticmethod
    def random_problem(n_dim: int, cities: int):
        cities = [City(n_dim) for _ in range(cities)]
        return TspProblem(cities)

    def random_solution(self) -> TspSolution:
        cpy = list(self.cities)
        random.shuffle(cpy)
        return TspSolution(self, cpy)

    @lru_cache(maxsize=int(2**16) )
    def evaluate(self, s: TspSolution):
        self.eval_count += 1
        return self.max_dist - self.route_distance(s.route)

    @staticmethod
    def route_distance(route: List[City]):
        pathDistance = 0
        path = route + [route[0]]
        for i in range(0, len(path) - 1):
            pathDistance += path[i].distance(path[i + 1])
        return pathDistance

    def __str__(self):
        return f"Tsp {len(self.cities)} cities & {self.n_dim} dim"



class TspSolution(Solution):

    def __init__(self, problem: TspProblem, route: List[City]):
        self.problem: TspProblem = problem
        self.route = route
        assert set(route) == set(problem.cities)

    @property
    def score(self):
        return self.problem.evaluate(self)


    def mutate(self, mutationRate: float) -> TspSolution:

        route = self.route

        for i in range(len(route)):
            if random.random() < mutationRate:
                j = int(random.random() * len(route))
                route[i], route[j] = route[j], route[i]

        return TspSolution(self.problem, route)



