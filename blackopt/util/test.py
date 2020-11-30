from typing import List, SupportsFloat, TypeVar, Type
from blackopt.abc import Solution, Problem


def test_definition(prob: Problem, sol_cls: Type[Solution]):
    """Test that a problem class and a solution class work together.

    Specifically, solution can be evaluated, mutated and crossed over.
    This function is designed to be called by users of the library on
    their problem and solution classes.
    """

    sol_cls.problem = prob
    s = sol_cls.random_solution()
    assert isinstance(s, sol_cls)
    assert isinstance(s.score, SupportsFloat)

    s2 = s.mutate(0.5)
    assert isinstance(s2, sol_cls)
    assert isinstance(s2.score, SupportsFloat)

    s3 = s.crossover(s2)[0]
    assert isinstance(s3, sol_cls)
    assert isinstance(s3.score, SupportsFloat)






