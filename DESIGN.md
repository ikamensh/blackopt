# Design

## High-Level Goal

Make blackopt the easiest way to experiment with, compare, and deploy metaheuristic optimization algorithms in Python.

The sharpest version:

> The fastest path from "I have a function to optimize" to "I know which algorithm works best and it's running in production."

## Three Pillars

### Experiment

It should be trivial to define a new problem, plug in a new algorithm, or tweak an existing one. More algorithms (particle swarm, differential evolution, CMA-ES), more benchmark problems (Rastrigin, Rosenbrock, real-world datasets), and better defaults so users get good results without tuning solver parameters.

### Compare

The comparison tooling is a genuine differentiator over alternatives like DEAP or scipy.optimize. Leaning into this means richer metrics, statistical significance testing, reproducible benchmarks, and publication-ready visualizations. The goal: run one function and get a defensible answer to "which algorithm works best for my problem?"

### Deploy

Bridging the gap from research toy to production use. Better documentation, a stable API, adaptive/auto-tuning solvers that pick good hyperparameters, and scaling to real problem sizes without friction.
