from __future__ import annotations
from collections import namedtuple

continuous_param = namedtuple('continuous', 'name min max')

import random
from numbers import Real

from typing import List, Union
from collections import deque

class Interval:
    def __init__(self, low: Real, high: Real,*, include_low=True, include_high=False):
        self.low = low
        self.high = high
        self.include_low = include_low
        self.include_high = include_high


    @property
    def length(self):
        return self.high - self.low

    def __contains__(self, number: Real):
        assert isinstance(number, Real)
        if self.include_low:
            if number == self.low:
                return True
        if self.include_high:
            if number == self.high:
                return True
        return self.low < number < self.high

    @staticmethod
    def split(interval: Interval, n: int) -> List[Interval]:
        split_len = interval.length / n

        children = [Interval(low=interval.low + i * split_len,
                             high=interval.low + (i+1) * split_len) for i in range(n)]

        children[0].include_low = interval.include_low
        children[-1].include_high = interval.include_high

        return children

    def sample(self):
        return random.random() * self.length + self.low

    def __add__(self, other: Interval):
        lower = min(self.low, other.low)
        higher = max(self.high, other.high)










class Optimizer:
    def __init__(self,
                 c_params: List[continuous_param],
                 buffer_len = 100_000):
        self.c_params = c_params
        self.xp_buffer = deque(maxlen=buffer_len)


    def sample(self):
        for c in self.c_params:
            return





