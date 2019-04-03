import random
import math
from matplotlib import pyplot as plt

# t = 0

optima = [0.1, 0.7]

max_chance = 1e-3
steep = 10

def chance_of_success(x):
    diff = min(abs(x - optimum_x) for optimum_x in optima)
    return max_chance * math.exp(-diff*10)

import numpy as np

args = np.linspace(0, 1, num=1000)
y = [chance_of_success(x) for x in args]

plt.plot(y)
plt.show()

from collections import namedtuple
trial = namedtuple('trial', 'x result')

samples = []
for i in range(int(1e6)):
    x = random.random()
    result = 1 if chance_of_success(x) > random.random() else 0
    samples.append( trial(x,result) )

naive_chance = sum([t.result for t in samples]) / len(samples)
print("Naive chance:",naive_chance)

successes = [t for t in samples if t.result]
maybe_good_x = sum([t.x for t in successes]) / len(successes)
print("Maybe good x:", maybe_good_x)

samples = []
for i in range(int(1e6)):
    result = 1 if chance_of_success(maybe_good_x) > random.random() else 0
    samples.append( trial(maybe_good_x,result) )

smart_chance = sum([t.result for t in samples]) / len(samples)
print("Smart chance:",smart_chance)
print(f"Improvement of {100*(smart_chance - naive_chance)/naive_chance:.2f} %")





