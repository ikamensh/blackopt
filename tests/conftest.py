import os
import sys

import pytest

cur_dir = os.path.dirname(__file__)
blackopt_dir = os.path.join(cur_dir, "..")
sys.path.append(blackopt_dir)

from blackopt.config import set_rootdir, get_rootdir
from blackopt.examples.problems.tsp import TspSolution, TspProblem
from blackopt.examples.problems.bumpy import BumpySolution, BumpyProblem
from blackopt.examples.problems.stepwise import StepSolution, StepProblem



@pytest.fixture()
def with_tmp_root(tmpdir):
    rootdir = get_rootdir()
    set_rootdir(tmpdir)
    yield
    set_rootdir(rootdir)


@pytest.fixture()
def tsp_problem():
    yield TspProblem.random_problem(5, 10)


@pytest.fixture(params=[
        (TspProblem, [5, 10], TspSolution),
        (BumpyProblem, [5, 10], BumpySolution),
        (StepProblem, [100], StepSolution),
    ])
def problem_solution_pair(request):

    prob_cls, params, sol_cls = request.param
    prob = prob_cls.random_problem(*params)
    sol_cls.problem = prob
    yield prob, sol_cls




