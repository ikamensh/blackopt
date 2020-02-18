from typing import List, SupportsFloat, TypeVar
from blackopt.abc import Solution, Problem


def test_definition(prob: Problem, sol_cls: TypeVar):

    s = sol_cls.random_solution()
    assert isinstance(s, sol_cls)
    assert isinstance(s.score, SupportsFloat)

    s2 = s.mutate(0.5)

    assert isinstance(s2, sol_cls)
    assert isinstance(s2.score, SupportsFloat)

    s3 = s.crossover(s2)[0]

    assert isinstance(s3, sol_cls)
    assert isinstance(s3.score, SupportsFloat)






