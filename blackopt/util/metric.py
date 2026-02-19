from __future__ import annotations

import bisect
import copy
from collections import defaultdict
from typing import DefaultDict, List, SupportsFloat


def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


class Metric:
    """Time-series metric for optimization curves. Supports merging via sum()."""

    def __init__(
        self,
        name: str = "unknown",
        x_label: str = "x",
        y_label: str = "y",
    ) -> None:
        self.name = name
        self.x_label = x_label
        self.y_label = y_label
        self.data: DefaultDict[float, List[float]] = defaultdict(list)

    @property
    def samples(self) -> int:
        """Number of y-values per x (assumes uniform across keys)."""
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))

    def add_record(self, x: SupportsFloat, y: SupportsFloat) -> None:
        self.data[float(x)].append(float(y))

    def _sort(self) -> None:
        self.data = defaultdict(list, dict(sorted(self.data.items())))

    def discard_warmup(self, part: float) -> None:
        """Remove the first `part` fraction of the x-range from data."""
        assert 0 < part < 1
        self._sort()
        keys = list(self.data.keys())
        if len(keys) < 2:
            return
        d_min, d_max = keys[0], keys[-1]
        span = d_max - d_min
        for k in keys:
            if k < d_min + part * span:
                del self.data[k]

    def _interpolate(self, key: float) -> float:
        """Linear interpolation for a key not in data."""
        keys = sorted(self.data.keys())
        if len(keys) == 1:
            return _avg(self.data[keys[0]])
        if key <= keys[0]:
            left, right = keys[0], keys[1]
        elif key >= keys[-1]:
            left, right = keys[-2], keys[-1]
        else:
            idx = bisect.bisect_right(keys, key)
            left, right = keys[idx - 1], keys[idx]
        val_left = _avg(self.data[left])
        val_right = _avg(self.data[right])
        slope = (val_right - val_left) / (right - left) if right != left else 0
        return val_left + slope * (key - left)

    def __add__(self, other: Metric) -> Metric:
        if self.samples == 0:
            return copy.deepcopy(other)
        if other.samples == 0:
            return copy.deepcopy(self)
        self._sort()
        other._sort()
        result = Metric(self.name, self.x_label, self.y_label)
        all_keys = sorted(set(self.data.keys()) | set(other.data.keys()))
        for k in all_keys:
            self_vals = self.data.get(k)
            if self_vals is None:
                self_vals = [self._interpolate(k)] * self.samples
            other_vals = other.data.get(k)
            if other_vals is None:
                other_vals = [other._interpolate(k)] * other.samples
            result.data[k] = self_vals + other_vals
        return result

    def __radd__(self, other: object) -> Metric:
        if other == 0:
            return self
        return NotImplemented
