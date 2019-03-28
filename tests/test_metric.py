from utils.plot import Metric
import pytest


def test_unique():
    m1 = Metric("x", "y")
    m2 = Metric("x", "y")

    assert not m1 is m2
    assert not m1 == m2

def test_summ_valid():
    m1 = Metric("x", "y")
    m1.add_many(2, [3,4,5])
    m1.add_many(3, [3,1])
    m1.add_many(4, [5])

    m2 = Metric("x", "y")
    m2.add_many(2, [3,4,15])
    m2.add_many(3, [3,10])
    m2.add_many(4, [7])

    m3 = m1 + m2

    assert len(m3.data.keys()) == 3

def test_summ():
    m1 = Metric("x", "y")
    m1.add_many(2, [3,4,5])

    m2 = Metric("x", "y")
    m2.add_many(2, [3,4,15])

    m3 = Metric("x", "y")
    m3.add_many(2, [3,4,15])

    m4 = sum( [m1,m2,m3] )

    assert 2 in m4.data.keys()

def test_summ_different_keys():
    m1 = Metric("x", "y")
    m1.add_many(2, [3,4,5])
    m1.add_many(3, [3,1])
    m1.add_many(4, [5])

    m2 = Metric("x", "y")
    m2.add_many(2, [3,4,15])
    m2.add_many(4, [7])

    with pytest.raises(AssertionError):
        m3 = m1 + m2

