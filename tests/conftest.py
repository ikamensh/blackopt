import pytest
from blackopt.config import set_rootdir, get_rootdir

from blackopt.examples.problems import TspSolution, TspProblem
from blackopt.examples.problems import BumpySolution, BumpyProblem
from blackopt.examples.problems import StepSolution, StepProblem

import sys

import os

cur_dir = os.path.dirname(__file__)
blackopt_dir = os.path.join(cur_dir, "..", "blackopt")
sys.path.append(blackopt_dir)

@pytest.fixture()
def with_tmp_root(tmpdir):
    rootdir = get_rootdir()
    set_rootdir(tmpdir)
    yield
    set_rootdir(rootdir)


@pytest.fixture()
def tsp_problem():
    yield TspProblem.random_problem(5, 10)

