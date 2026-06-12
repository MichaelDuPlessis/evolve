# evolve-rs (Python bindings)

Rust-powered evolutionary algorithms for Python. The core engine runs at native Rust speed; you supply a plain Python callable as the fitness function.

## Installation

```bash
pip install evolve-rs
```

For local development (requires [maturin](https://github.com/PyO3/maturin)):

```bash
maturin develop
```

## Quick Start — OneMax

```python
from evolve import EvolutionaryAlgorithm, MaxGenerations
from evolve.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
from evolve.initializers import RangedRandom

def onemax(genome):
    return float(sum(g > 127 for g in genome))

ea = EvolutionaryAlgorithm(
    initializer=RangedRandom(min_len=20, max_len=20),
    operators=Fill(
        Pipeline([
            Combine([Tournament(3), Tournament(3)]),
            SinglePoint(),
            RandomReset(),
        ])
    ),
    fitness=onemax,
    termination=MaxGenerations(200),
    population_size=100,
)

result = ea.run()
best = result.best()
print(f"Best fitness: {best.fitness}  genome[:5]: {best.genome[:5]}")
```

## API Overview

### `evolve`

| Class | Description |
|---|---|
| `EvolutionaryAlgorithm(initializer, operators, fitness, termination, population_size, comparator=None, seed=None)` | Main entry point. |
| `MaxGenerations(n)` | Stop after `n` generations. |

### `evolve.operators`

| Class | Description |
|---|---|
| `Tournament(size)` | Tournament selection. |
| `SinglePoint()` | Single-point crossover. |
| `RandomReset()` | Random-reset mutation. |
| `Pipeline([op, ...])` | Apply operators sequentially. |
| `Combine([op, ...])` | Run operators on the same input and merge results. |
| `Fill(op)` | Repeat `op` until population is full. |
| `FillFixed(op, size)` | Repeat `op` until exactly `size` individuals. |

### `evolve.initializers`

| Class | Description |
|---|---|
| `RangedRandom(min_len, max_len)` | Initialize genomes with random `u8` values and random length in `[min_len, max_len]`. |

### `evolve` comparators (optional)

Pass `comparator=Maximize()` (default) or `comparator=Minimize()` to `EvolutionaryAlgorithm`.

### Results

`ea.run()` returns a `RunResult`:

| Attribute / Method | Type | Description |
|---|---|---|
| `result.best()` | `Individual` | Best individual found. |
| `result.population` | `list[Individual]` | Final population. |
| `result.generations` | `int` | Number of generations run. |
| `result.total_duration` | `float` | Wall time in seconds. |
| `result.best_fitness` | `list[float]` | Best fitness per generation. |
| `result.generation_durations` | `list[float]` | Duration per generation (seconds). |

`Individual` has `genome: list[int]` (values 0–255) and `fitness: float`.

## Fitness Functions

Any Python callable works as a fitness function:

```python
fitness=lambda genome: float(sum(genome))
```

The function receives a `list[int]` (u8 values cast to int) and must return a `float`.

## Performance

Built-in operators (selection, crossover, mutation) execute entirely in Rust without touching the Python interpreter. The fitness callback crosses the Python/Rust boundary once per individual per generation — this is the main source of overhead compared to pure-Rust usage.

## Genome Type

Genomes are `Vec<u8>` (unsigned 8-bit integers, values 0–255). Additional genome types are planned for future phases.

## Rust Crate

For native Rust usage see the [evolve crate](../README.md).
