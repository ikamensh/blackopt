import os
import sys

import pytest

cur_dir = os.path.dirname(__file__)
blackopt_dir = os.path.join(cur_dir, "..")
sys.path.append(blackopt_dir)

from blackopt.config import set_rootdir, get_rootdir
from blackopt.examples.problems import TspSolution, TspProblem



@pytest.fixture()
def with_tmp_root(tmpdir):
    rootdir = get_rootdir()
    set_rootdir(tmpdir)
    yield
    set_rootdir(rootdir)


@pytest.fixture()
def tsp_problem():
    yield TspProblem.random_problem(5, 10)

