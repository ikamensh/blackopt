import pytest

from blackopt.persistence import ContinuousOptimizer
from blackopt.exceptions import BlackoptException
from blackopt.examples.problems import TspSolution
from blackopt.algorithms import RandomSearch

@pytest.mark.timeout(0.1)
def test_stops_on_same_error(with_tmp_root, tsp_problem):

    solver = RandomSearch(tsp_problem, TspSolution)

    def broken_solve(*args, **kwargs):
        raise Exception("Broken beyond repair")

    solver.solve = broken_solve

    co = ContinuousOptimizer(solver)
    with pytest.raises(BlackoptException):
        co.run()


@pytest.mark.timeout(0.1)
def test_tolerant_to_occasional_errors(with_tmp_root, tsp_problem):

    solver = RandomSearch(tsp_problem, TspSolution)
    pytest_ctr = 0

    def fake_solve(self, *args, **kwargs):
        nonlocal pytest_ctr
        if pytest_ctr >= 5:
            raise Exception("Broken beyond repair")
        else:
            pytest_ctr += 1
            if not pytest_ctr % 2:
                raise Exception("Something happened")


    RandomSearch.solve = fake_solve

    co = ContinuousOptimizer(solver)
    with pytest.raises(BlackoptException):
        co.run()

    assert pytest_ctr == 5


@pytest.mark.timeout(0.1)
def test_termination_condition(with_tmp_root, tsp_problem):

    solver = RandomSearch(tsp_problem, TspSolution)
    pytest_ctr = 0

    def fake_solve(self, *args, **kwargs):
        nonlocal pytest_ctr
        pytest_ctr += 1


    RandomSearch.solve = fake_solve

    def termination_condition(self):
        nonlocal pytest_ctr
        if pytest_ctr and not pytest_ctr % 2:
            return True

    co = ContinuousOptimizer(solver, termination_condition=termination_condition)
    co.run()

    assert pytest_ctr == 2


