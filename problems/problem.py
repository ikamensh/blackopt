from __future__ import annotations
import abc
from typing import List


class Solution(abc.ABC):
    score : float = None
    problem : Problem = None

    @abc.abstractmethod
    def mutate(self, rate: float) -> Solution:
        raise NotImplementedError()

    # @abc.abstractmethod
    # def crossover(self, other: Solution) -> List[Solution]:
    #     raise NotImplementedError()


class Problem(abc.ABC):

    eval_count: int = None

    @abc.abstractmethod
    def evaluate(self, s: Solution) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def random_solution(self) -> Solution:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()




