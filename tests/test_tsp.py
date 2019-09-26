from blackopt.examples.problems import TspSolution, TspProblem
from blackopt.examples.problems import BumpySolution, BumpyProblem
from blackopt.examples.problems import StepSolution, StepProblem

import pytest


@pytest.mark.parametrize(
    ("prob_cls", "prob_args", "sol_cls"),
    [
        (TspProblem, [5, 10], TspSolution),
        (BumpyProblem, [5, 10], BumpySolution),
        (StepProblem, [100], StepSolution),
    ],
)
def test_eval_count(prob_cls, prob_args, sol_cls):

    prob = prob_cls.random_problem(*prob_args)

    sol_cls.problem = prob
    sol = sol_cls.random_solution()

    score = sol.score
    score2 = sol.score

    assert prob.eval_count == 1

import math

@pytest.mark.parametrize(
    ("prob_cls", "prob_args", "sol_cls"),
    [
        (TspProblem, [50, 50], TspSolution),
        (BumpyProblem, [50, 50], BumpySolution),
        (StepProblem, [100], StepSolution),
    ],
)
def test_similarity(prob_cls, prob_args, sol_cls):
    prob = prob_cls.random_problem(*prob_args)

    sol_cls.problem = prob
    sol = sol_cls.random_solution()
    sol2 = sol_cls.random_solution()


    assert math.isclose(sol.similarity(sol), 1)
    assert not math.isclose(sol.similarity(sol), sol.similarity(sol2))

