import math
from blackopt.util import test


def test_eval_count(problem_solution_pair):

    prob, sol_cls = problem_solution_pair

    sol = sol_cls.random_solution()

    score = sol.score
    score2 = sol.score

    assert prob.eval_count == 1


def test_similarity(problem_solution_pair):

    prob, sol_cls = problem_solution_pair

    sol = sol_cls.random_solution()
    sol2 = sol_cls.random_solution()


    assert math.isclose(sol.similarity(sol), 1)
    assert not math.isclose(sol.similarity(sol), sol.similarity(sol2))

def test_definitions(problem_solution_pair):
    prob, sol_cls = problem_solution_pair
    test.test_definition(prob, sol_cls)
