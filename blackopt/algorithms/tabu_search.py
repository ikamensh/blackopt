from collections import deque
from typing import Type

from blackopt.abc import Problem, Solution, Solver


class TabuSearch(Solver):
    """Single-trajectory search with a tabu list to avoid cycles and local optima."""

    name = "Tabu Search"

    def __init__(
        self,
        problem: Problem,
        solution_cls: Type[Solution],
        mutation_rate: float,
        tabu_size: int = 10,
        tabu_threshold: float = 0.9,
        n_candidates: int = 5,
    ):
        assert 0 < mutation_rate <= 1
        assert tabu_size > 0
        assert 0 <= tabu_threshold <= 1
        assert n_candidates >= 1

        self.mutation_rate = mutation_rate
        self.tabu_size = tabu_size
        self.tabu_threshold = tabu_threshold
        self.n_candidates = n_candidates

        super().__init__(problem, solution_cls)

        self.current_solution = self.best_solution
        self.tabu_list: deque[Solution] = deque(maxlen=tabu_size)

    def _is_tabu(self, candidate: Solution) -> bool:
        if not self.tabu_list:
            return False
        return any(
            candidate.similarity(tabu) >= self.tabu_threshold
            for tabu in self.tabu_list
        )

    def step(self):
        candidates = [
            self.current_solution.mutate(self.mutation_rate)
            for _ in range(self.n_candidates)
        ]
        candidates.sort(key=lambda s: s.score, reverse=True)

        accepted = None
        for c in candidates:
            if self._is_tabu(c):
                if c.score > self.best_solution.score:
                    accepted = c
                    break
            else:
                accepted = c
                break

        if accepted is not None:
            self.tabu_list.append(self.current_solution)
            self.current_solution = accepted
            if accepted.score > self.best_solution.score:
                self.best_solution = accepted

        self.record()

    def __str__(self):
        return (
            f"{self.name} mr-{self.mutation_rate} "
            f"tabu-{self.tabu_size} thresh-{self.tabu_threshold}"
        )
