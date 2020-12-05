from unittest.mock import patch

import pytest

from blackopt import EarlyStopException
from blackopt.algorithms import OffspringSelection
from blackopt.examples.problems.bumpy import BumpyProblem, BumpySolution


def test_step():
    """Test that a GAOS solver can make a step and that population changes in this process. """
    prob = BumpyProblem.random_problem(5, 5)
    ga = OffspringSelection(prob, BumpySolution, popsize=30, mutation_rate=1)
    before = set(ga.population)
    ga.step()

    assert set(ga.population) != before

@patch.object(OffspringSelection, "_breed", autospec=True)
def test_terminates_if_cant_breed(fake_breed):

    prob = BumpyProblem.random_problem(5, 5)
    ga = OffspringSelection(prob, BumpySolution, popsize=30, mutation_rate=1)
    fake_breed.return_value = []

    with pytest.raises(EarlyStopException):
        ga.step()
