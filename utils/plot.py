from __future__ import annotations
from dataclasses import dataclass, field
from utils.running_avg import apply_running_average
import statistics
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

import os




def maybe_make_dir(folder):
    try:
        os.makedirs(folder)
    except BaseException:
        pass


@dataclass(eq=False)
class Metric:
    x_label: str
    y_label: str
    data: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list))

    def add_record(self, x: int, y: float):
        self.data[x].append(y)

    def add_many(self, x: int, ys: List[float]):
        self.data[x].extend(ys)

    def __add__(self, other: Metric):
        assert self.data.keys() == other.data.keys()

        result = copy.copy(self)
        for k, v in other.data.items():
            result.add_many(k, v)

        return result

    def __radd__(self, other):
        # support sum
        assert other is 0
        return self


def ez_plot(metric: Metric, folder, name=None):

    maybe_make_dir(folder)
    plt.clf()

    avg = np.array([sum(l) / len(l) for l in metric.data.values()])
    stdev = []
    for l in metric.data.values():
        if len(l) > 1:
            stdev.append(statistics.stdev(l))
        else:
            stdev.append(0)
    stdev = np.array(stdev)

    plt.plot(metric.data.keys(), avg)
    plt.fill_between(metric.data.keys(), avg - stdev, avg + stdev, alpha=0.2)
    plt.xlabel(metric.x_label)
    plt.ylabel(metric.y_label)
    plt.grid()

    path = os.path.join(
        folder,
        (name or f"{metric.y_label}_{metric.x_label}") +
        ".png")
    print(path)
    plt.savefig(path)


def plot_group(metrics: Dict[str, Metric], folder: str, name: str = None):

    matplotlib.rcParams.update({'font.size': 3})

    maybe_make_dir(folder)
    plt.clf()

    metric = list(metrics.values())[0]
    plt.xlabel(metric.x_label)
    plt.ylabel(metric.y_label)
    plt.grid()

    for label, metric in metrics.items():

        avg = np.array([sum(l) / len(l) for l in metric.data.values()])
        stdev = []
        for l in metric.data.values():
            if len(l) > 1:
                stdev.append(statistics.stdev(l))
            else:
                stdev.append(0)
        stdev = np.array(stdev)

        smoothen = len(avg) / (len(avg) + 100)

        avg = apply_running_average(avg, smoothen)
        stdev = apply_running_average(stdev, smoothen)

        plt.plot(metric.data.keys(), avg, label=label, linewidth=0.65)
        plt.fill_between(
            metric.data.keys(),
            avg - stdev,
            avg + stdev,
            alpha=0.2)

    plt.legend(loc='best')
    path = os.path.join(
        folder,
        (name or f"{metric.y_label}_{metric.x_label}") +
        ".png")
    print(path)
    plt.savefig(path, dpi=300)


if __name__ is "__main__":
    m = Metric('x', 'y')
    m.add_many(1, [1, 2, 3])
    m.add_record(2, 4)
    m.add_many(4, [5, 6, 12])

    m2 = Metric('x', 'y')
    m2.add_many(1, [4, 3, 5])
    m2.add_record(2, 5)
    m2.add_record(2, 9)
    m2.add_many(4, [5, 6, -1])

    from config import root_dir
    plot_group({'m1': m, 'm2': m2}, root_dir, "bla3")


# def plot_many(name, folder, *args):
#     maybe_make_dir(folder)
#     plt.clf()
#     for array in args:
#         plt.plot(array)
#     plt.ylabel(name)
#     plt.xlabel('Generation')
#     plt.grid()
#     plt.savefig(os.path.join(folder, name + ".png"))


def plot_histogram(array, name, folder):
    maybe_make_dir(folder)
    plt.clf()
    plt.hist(array)
    plt.savefig(os.path.join(folder, name + ".png"))
