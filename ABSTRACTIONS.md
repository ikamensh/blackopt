# Abstractions

## The Core Idea

The Problem/Solution split answers two different questions:

- **Problem**: "Given a candidate, how good is it?" — one method: `evaluate(solution) → float`
- **Solution**: "What does a candidate look like, and how do I make new ones?" — three methods: `random_solution`, `mutate`, `crossover`

Solvers never need to know what's inside either one. They just call "give me a random candidate", "make a variant", "combine two", and "score this" — and the optimization loop works.

## The Contract

To use blackopt, you answer two questions:

**"What am I optimizing?"** — Write a Problem with one method: `evaluate(solution) → float`. Higher is better.

**"What do candidates look like?"** — Write a Solution with three methods:
- `random_solution()` — generate one from scratch
- `mutate(rate)` — return a slightly different copy
- `crossover(other)` — blend two into children

That's the whole contract. Every solver in the library works with any Problem/Solution pair that fulfills it.

## Design Choices That Might Surprise You

### Solution carries the transformation logic, not the Solver

In many frameworks the solver handles mutation and crossover. Here, the Solution knows how to transform itself because the right way to mutate depends entirely on the representation. Swapping cities in a route (TSP) is nothing like flipping a float in a vector. This is why the same `EvolutionaryAlgorithm` solves both a list-of-floats problem and a permutation problem without any changes.

### `problem` is a class variable on Solution

All solutions of a given type share the same problem. You set `MySolution.problem = my_problem` once before solving, and every instance knows how to score itself. This keeps the per-solution API clean — no passing `problem` into every constructor — but it means you can't have two problems using the same Solution class simultaneously.

### Score is lazy and cached

`solution.score` doesn't evaluate until first access, then caches forever. Solutions are immutable — `mutate` and `crossover` return new instances, never modify in place. This means the evaluation count tracks real work, not redundant re-evaluations.

## Concrete Example: StepProblem

The simplest problem in the library. A solution is a list of floats. Fitness is how many exceed their threshold.

```python
class StepProblem(Problem):
    def evaluate(self, s):
        return sum(1 for val, t in zip(s.genes, self.thresholds) if val > t)

class StepSolution(Solution):
    def random_solution():
        return StepSolution([random.random() for _ in range(problem.n_dim)])

    def mutate(self, rate):
        new = [random.random() if random.random() < rate else v for v in self.genes]
        return StepSolution(new)

    def crossover(self, other):
        point = random.randint(1, len(self.genes) - 1)
        return [StepSolution(self.genes[:point] + other.genes[point:]),
                StepSolution(other.genes[:point] + self.genes[point:])]
```

## The "Aha" Moment

Now look at TSP — a completely different representation (permutation of cities instead of a float vector). Mutation swaps cities. Crossover preserves order. But the Solver interface is identical:

```python
class TspSolution(Solution):
    def random_solution():
        route = list(problem.cities)
        random.shuffle(route)
        return TspSolution(route)

    def mutate(self, rate):
        route = list(self.route)
        for i in range(len(route)):
            if random.random() < rate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return TspSolution(route)

    def crossover(self, other):
        point = random.randint(1, len(self.route) - 1)
        left = self.route[:point]
        right = [c for c in other.route if c not in left]
        return [TspSolution(left + right)]
```

The same `EvolutionaryAlgorithm`, `OffspringSelection`, or `Sasegasa` works on both without modification. That's the payoff of putting transformation logic in the Solution.
