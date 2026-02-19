# blackopt

A Python black-box optimization framework for solving problems where you can evaluate a fitness function but don't have access to its internal structure (no gradients, no closed-form).

## Architecture

blackopt follows a clean **Problem - Solution - Solver** abstraction:

- **Problem** defines the fitness landscape (what to optimize)
- **Solution** represents a candidate (supports mutation, crossover, random generation)
- **Solver** implements the search strategy

## Algorithms

| Algorithm | Description |
|---|---|
| Random Search | Baseline random sampling, with multicore variant |
| Hill Climbing | Local search, accepts only improvements |
| Simulated Annealing | Escapes local optima via probabilistic acceptance |
| Genetic Algorithm | Population-based with selection, crossover, mutation, elitism |
| Offspring Selection | GA variant where offspring must beat parents |
| RAPGA | Offspring Selection + diversity enforcement (removes similar individuals) |
| SASEGASA | Multi-village speciation GA with parallel evolution |
| Multicore GA | Parallel populations with periodic migration |

## Example Problems

- **StepProblem** - multi-dimensional threshold-based benchmark
- **BumpyProblem** - smooth landscape with many local optima
- **TspProblem** - Traveling Salesman Problem (combinatorial)

## Use Cases

- **Black-box optimization** - when you can only evaluate, not differentiate
- **Algorithm benchmarking** - built-in `compare_solvers()` for fair comparisons with visualization
- **Combinatorial optimization** - TSP, scheduling, routing
- **Hyperparameter tuning** - treat any tuning problem as a black box
- **Research** - easily extend with new solvers, problems, or solution representations
- **Long-running optimization** - checkpointing and resumption via `ContinuousOptimizer`

## Installation

```
pip install blackopt
```

## Testing

```bash
pytest                 # default: fast suite (~0.4s, 60 tests)
pytest -m slow         # slow autosolve integration tests (~80s)
pytest -m ""           # full suite (all tests)
```

Fast tests include mocked autosolve (Result structure, portfolio building) and algorithm unit tests.

## Dependencies

- `numpy` - numerical operations
- `pathos` - multicore process pools
- `ilya-ezplot` - metrics and plotting
- `structlog` - structured logging
