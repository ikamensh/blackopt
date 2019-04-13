from __future__ import annotations
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List, Any, Dict

from utils.plot.metric.interpolate import merge


@dataclass(eq=False)
class Metric:
    x_label: str
    y_label: str
    data: DefaultDict[float, List[float]] = field(default_factory=lambda: defaultdict(list))
    style_kwargs: Dict[str: Any] = field(default_factory=lambda: {})

    def add_record(self, x: float, y: float):
        self.data[x].append(y)

    def add_many(self, x: float, ys: List[float]):
        self.data[x].extend(ys)

    def __add__(self, other: Metric):
        merge(self.data, other.data)


        result = Metric(self.x_label, self.y_label)
        result.data = defaultdict(list)
        result.data.update( {k:v for k,v in sorted(self.data.items())})
        for k, v in other.data.items():
            result.add_many(k, v)

        return result

    def __radd__(self, other):
        # support sum
        assert other is 0
        return self


