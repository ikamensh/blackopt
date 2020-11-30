from typing import List
import random
import math
from functools import lru_cache

from blackopt.abc import Problem, Solution
from blackopt.examples.problems.tsp.city import City



class TspProblem(Problem):
    """Traveling salesman problem. The problem is NP-hard.

    An instance of a problem will have `n_dim` dimensions and `len(cities)` cities. A city is
    a point in the N-dimensional space. A solution to TSP is a route that visits all cities
    exactly once. The simplest representation of such a route is a sequence of cities.
    """
    name = "TSP"

    def __init__(self, cities: List[City]):
        self.cities = cities
        self.n_dim = len(cities[0].coordinates)
        self.eval_count = 0
        self.score_span = self.max_dist = len(self.cities) * math.sqrt(self.n_dim)

    @staticmethod
    def random_problem(n_dim: int, cities: int):
        cities = [City(n_dim) for _ in range(cities)]
        return TspProblem(cities)

    def evaluate(self, s: 'TspSolution'):
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
        return f"TSP with {len(self.cities)} cities in {self.n_dim}-dim space."


class TspSolution(Solution):
    problem: TspProblem

    def __init__(self, route: List[City]):
        self.route = route
        assert set(route) == set(self.problem.cities)

    @staticmethod
    def random_solution() -> 'TspSolution':
        """ A random solution to a TSP is a random sequence of all its cities. """
        cpy = list(TspSolution.problem.cities)
        random.shuffle(cpy)
        return TspSolution(cpy)

    def mutate(self, mutationRate: float) -> 'TspSolution':
        """For each city in the route, swap it with a random other city with given probability. """
        route = list(self.route)

        for i in range(len(route)):
            if random.random() < mutationRate:
                j = int(random.random() * len(route))
                route[i], route[j] = route[j], route[i]

        return TspSolution(route)

    def crossover(self, other: 'TspSolution'):
        crossover_point = random.randint(1, len(self.route) - 1)

        child_left = self.route[:crossover_point]
        check = {city.uid for city in child_left}
        child_right = [city for city in other.route if city.uid not in check]

        return [TspSolution(child_left + child_right)]

    @lru_cache(maxsize=1>>16)
    def similarity(self, other: 'TspSolution'):
        r1 = self.route
        r2 = other.route

        moves_1 = { (r1[i], r1[i+1]) for i in range(len(r1)-1)}
        moves_1.add( (r1[-1], r1[0]) )
        moves_2 = { (r2[i], r2[i+1]) for i in range(len(r2)-1)}
        moves_2.add((r2[-1], r2[0]))

        return len( moves_1.intersection(moves_2) ) / len(moves_1)
