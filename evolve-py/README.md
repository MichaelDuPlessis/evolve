# evolve-rs (Python bindings)

Rust-powered evolutionary algorithms for Python. The core engine runs at native Rust speed; you supply plain Python callables for fitness, operators, and more.

## Installation

```bash
pip install evolve-rs
```

```python
import evolve  # Note: import as 'evolve', not 'evolve_rs'
```

For local development (requires [maturin](https://github.com/PyO3/maturin)):

```bash
maturin develop
```

## Quick Start — OneMax

```python
from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
from evolve.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
from evolve.initializers import RangedRandom

def onemax(genome):
    return float(sum(g > 127 for g in genome))

ea = EvolutionaryAlgorithm(
    initializer=RangedRandom(min_len=20, max_len=20, dtype="u8"),
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
    comparator=Maximize(),
)

result = ea.run()
best = result.best()
print(f"Best fitness: {best.fitness}  genome[:5]: {best.genome[:5]}")
```

## Genome Types (`dtype`)

| dtype | Python type | Range |
|---|---|---|
| `u8` (default) | `int` | 0–255 |
| `u16` | `int` | 0–65535 |
| `u32` | `int` | 0–4294967295 |
| `u64` | `int` | 0–2^64-1 |
| `i8` | `int` | -128–127 |
| `i16` | `int` | -32768–32767 |
| `i32` | `int` | -2^31–2^31-1 |
| `i64` | `int` | -2^63–2^63-1 |
| `f32` | `float` | IEEE 754 single |
| `f64` | `float` | IEEE 754 double |

## API Reference

### `evolve`

```python
EvolutionaryAlgorithm(initializer, operators, fitness, termination, population_size, comparator=None, seed=None)
```

- `run()` → `RunResult`
- `run_with(collector)` → runs EA and calls `collector.on_generation(gen, best_fitness, population)` each generation

```python
MaxGenerations(n)   # stop after n generations
Maximize()          # higher fitness is better (default)
Minimize()          # lower fitness is better
```

### `evolve.initializers`

| Class | Description |
|---|---|
| `RangedRandom(min_len, max_len, dtype="u8")` | Random genomes with length in `[min_len, max_len]` |
| `Random(genome_length, dtype="u8")` | Random genomes of fixed length |

Custom initializer callable: receives `population_size: int`, returns `list[list[int|float]]`.

### `evolve.operators` — Sequential

| Class | Description |
|---|---|
| `Tournament(size)` | Tournament selection |
| `Elitism(amount=1)` | Keep best individuals |
| `RouletteWheel()` | Fitness-proportionate selection |
| `Rank()` | Rank-based selection |
| `Sus(count)` | Stochastic universal sampling |
| `SinglePoint()` | Single-point crossover |
| `TwoPoint()` | Two-point crossover |
| `Uniform()` | Uniform crossover |
| `Arithmetic()` | Arithmetic crossover (f32/f64 only) |
| `RandomReset()` | Random-reset mutation |
| `Swap()` | Swap mutation |
| `Inversion()` | Inversion mutation |
| `Scramble()` | Scramble mutation |
| `Creep(step)` | Creep mutation (integer dtypes only) |
| `Gaussian(std_dev)` | Gaussian mutation (f32/f64 only) |
| `SegmentDuplication(fraction, max_len)` | Duplicate a random segment |
| `SegmentDeletion(fraction, min_len)` | Delete a random segment |
| `Fill(op)` | Repeat op until population full |
| `FillFixed(op, size)` | Repeat op until exactly `size` individuals |
| `Pipeline([ops])` | Apply operators sequentially |
| `Combine([ops])` | Run each op on same input, merge results |
| `Weighted([(op, weight)])` | Weighted combination |
| `Proportional([(op, weight)], size=None)` | Proportional combination |
| `Repeat(op, n)` | Apply op n times |
| `Identity()` | Pass-through (no-op) |
| `WithRate(op, rate)` | Apply op with given probability |
| `Conditional(predicate, if_true, if_false)` | Choose op based on `predicate(generation, pop_size)` |

### `evolve.parallel` — Parallel Operators

| Class | Description |
|---|---|
| `ParallelFill(op, target_size)` | Fill population in parallel |
| `ParallelRandomReset()` | Parallel random-reset mutation |
| `ParallelSwap()` | Parallel swap mutation |
| `ParallelInversion()` | Parallel inversion mutation |
| `ParallelScramble()` | Parallel scramble mutation |
| `ParallelCreep(step)` | Parallel creep (integer dtypes only) |
| `ParallelGaussian(std_dev)` | Parallel Gaussian (f32/f64 only) |

## Custom Callbacks

### Custom operator

```python
def my_operator(population, generation):
    # population: list of {"genome": [...], "fitness": float|None}
    # generation: int
    return [ind["genome"] for ind in population]
```

### Custom termination

```python
def my_termination(generation, best_fitness):
    return best_fitness >= 100.0 or generation >= 500
```

### Custom comparator

```python
def my_comparator(a, b):
    return a > b  # True if a is better than b
```

### Custom initializer

```python
def my_initializer(population_size):
    return [[0] * 20 for _ in range(population_size)]
```

### Collector (`run_with`)

```python
class MyCollector:
    def on_generation(self, generation: int, best_fitness: float, population: list):
        print(f"Gen {generation}: best={best_fitness}")

result = ea.run_with(MyCollector())
```

## Grammatical Evolution (GE)

```python
from evolve import Grammar, StandardMapper, GeFitness

grammar = (Grammar.builder()
    .rule("expr", [["expr", "+", "term"], ["term"]])
    .rule("term", [["0"], ["1"]])
    .start("expr")
    .build())

mapper = StandardMapper(max_wraps=3)
fitness = GeFitness(grammar, mapper, lambda s: float(eval(s)), penalty=0.0)
```

## Experiment Runner

```python
from evolve import Experiment

def factory():
    return EvolutionaryAlgorithm(...)

experiment = Experiment(factory, trials=10)
results = experiment.run()  # list of RunResult
```

## Results

`RunResult` attributes: `population`, `generations`, `total_duration`, `best_fitness`, `generation_durations`  
`RunResult.best()` → `Individual` with `.genome` and `.fitness`

## Rust Crate

For native Rust usage see the [evolve crate](../README.md).
