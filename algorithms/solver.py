import abc
from problems.problem import Problem, Solution


class Solver(abc.ABC):
    name: str = None
    best_solution: Solution = None
    problem: Problem = None

    @abc.abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.best_solution: Solution = self.problem.random_solution()
        self.problem.eval_count = 0

    def __str__(self):
        return str(self.name)
