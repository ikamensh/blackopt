import random

from blackopt.algorithms import EvolutionaryAlgorithm

from blackopt.distributed.network.client import ClientSocket


def start_worker():
    client = ClientSocket()
    problem, solution_cls = client.connect()
    solver = EvolutionaryAlgorithm(problem, solution_cls, 100, 1)

    while True:

        solver.step()
        client.send(solver.best_solution, random.choice(solver.population))

        new = client.receive()
        solver.population.append(new)
        solver._rank()
        if not solver.generation % 10:
            print(solver.generation, solver.best_solution.score)


start_worker()