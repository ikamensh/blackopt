from typing import DefaultDict, List, Tuple
import bisect

metric_key = float
metric_data = DefaultDict[metric_key, List[float]]
def avg(a): return sum(a) / len(a)

def find_closest(series: metric_data, key: metric_key) -> Tuple[metric_key, metric_key]:
    """ assumes keys are sorted."""
    keys = list(series.keys())
    if min(keys) > key: # neet to extrapolate
        left, right = keys[:2]
    elif max(keys) < key:
        left, right = keys[-2:]
    else:
        idx = bisect.bisect(keys, key)
        left, right = keys[idx-1], keys[idx]

    return left, right



def complete(series: metric_data, key: metric_key) -> None:
    """ uses interpolation or extrapolation to insert an intermediate value into mapping """
    if key in series:
        return
    else:
        left, right = find_closest(series, key)
        val_left, val_right = avg(series[left]), avg(series[right])
        slope = (val_right - val_left) / (right - left)
        dx = key - left

        series[key].append(val_left + slope * dx)


def merge(a: metric_data, b: metric_data) -> None:
    """ Modifies metric data a and b looking for keys not shared between the two,
     and inserting interpolated values"""
    all_keys = set(a.keys()) | set(b.keys())

    for k in all_keys:
        complete(a, k)
        complete(b, k)
