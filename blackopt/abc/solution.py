import abc
from typing import TYPE_CHECKING, List, Dict, SupportsFloat

if TYPE_CHECKING:
    from blackopt.abc import Problem


class Solution(abc.ABC):
    # NOTE: ``problem`` is a *class* variable shared by all instances of a
    # concrete Solution subclass.  It is set by ``Solver.__init__`` via
    # ``solution_cls.problem = problem``.  In multicore / multiprocess
    # scenarios each worker must re-assign ``solution_cls.problem`` after
    # the fork (the ``iteration()`` helpers in sasegasa modules already do
    # this).  Converting it to an instance attribute would require changing
    # every ``random_solution()`` static method to a classmethod, so we keep
    # the class-level assignment and guard against accidental ``None`` access
    # via the ``score`` property below.
    problem: 'Problem' = None

    # ``_score`` and ``regularization_score`` are declared as *class*
    # attributes with default values.  Python's attribute-lookup semantics
    # mean that these defaults act as instance-level values until an
    # instance explicitly assigns to them (at which point the instance gets
    # its own copy in ``__dict__``).  This is safe for ``_score`` because
    # the ``score`` property always writes ``self._score`` on first access,
    # and for ``regularization_score`` because algorithms that modify it
    # (e.g. Rapga) always assign to ``individual.regularization_score``.
    _score: float = None
    regularization_score = 0

    @staticmethod
    @abc.abstractmethod
    def random_solution() -> 'Solution':
        raise NotImplementedError()

    @abc.abstractmethod
    def mutate(self, rate: float) -> 'Solution':
        raise NotImplementedError()

    @abc.abstractmethod
    def crossover(self, other: 'Solution') -> List['Solution']:
        raise NotImplementedError()

    @property
    def score(self):
        if self._score is None:
            if self.problem is None:
                raise RuntimeError(
                    f"{type(self).__name__}.problem is not set. "
                    "Create a Solver instance first, or assign "
                    "SolutionSubclass.problem = problem manually."
                )
            self._score = self.problem.evaluate(self)
        return self._score

    def similarity(self, other: 'Solution'):
        if other is self or other == self:
            return 1
        else:
            return 0

    def metrics(self) -> Dict[str, SupportsFloat]:
        return {"score": self.score}
