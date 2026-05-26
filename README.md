# evolve

[![Crates.io](https://img.shields.io/crates/v/evolve.svg)](https://crates.io/crates/evolve)
[![Documentation](https://img.shields.io/docsrs/evolve)](https://docs.rs/evolve)
[![License](https://img.shields.io/crates/l/evolve.svg)](LICENSE)

A generic, composable genetic algorithm framework for Rust.

> **Note:** This library is in rapid development. While breaking changes will be avoided where possible, there is no guarantee of API stability until 1.0.

`evolve` provides the building blocks to assemble genetic algorithms from reusable, type-safe components. Operators are composed using combinators — chain them into pipelines, weight them probabilistically, or repeat them to fill a population — all with zero-cost abstractions.

## Features

- Fully generic over genome type, fitness type, RNG, and fitness comparator
- Built-in operators for selection, crossover, and mutation
- Composable combinators for structuring the flow of the algorithm
- `Maximize` and `Minimize` fitness comparators out of the box
- Closures work as fitness evaluators and comparators via blanket trait impls
- Minimal dependencies: `rand` and `vecpool` (optional `pooled` for parallel execution)
- Optional `serde` support for serializing populations and results

## Available Operators

### Selection
- **Tournament** — tournament selection with replacement
- **Elitism** — preserve the best N individuals
- **RouletteWheel** — fitness-proportionate selection
- **Rank** — rank-based selection (avoids premature convergence)
- **Sus** — stochastic universal sampling (low-variance fitness-proportionate)

### Crossover
- **SinglePoint** — single-point crossover
- **TwoPoint** — two-point crossover
- **Uniform** — per-gene random parent selection
- **Arithmetic** — numeric blending for continuous genomes (f32/f64)

### Mutation
- **RandomReset** — replace a random gene with a new random value
- **Gaussian** — add Gaussian noise (continuous genomes)
- **Creep** — add a small random offset (discrete genomes)
- **Swap** — swap two random genes (permutation problems)
- **Inversion** — reverse a random segment (permutation problems)
- **Scramble** — shuffle a random segment (permutation problems)
- **SegmentDuplication** — duplicate a random segment (variable-length)
- **SegmentDeletion** — delete a random segment (variable-length)

### Combinators
- **Pipeline** — chain operators sequentially
- **Fill** — repeat an operator until target population size
- **Combine** — run operators on same input, merge outputs
- **Weighted** — probabilistically select one operator per call
- **Repeat** — apply an operator N times
- **Proportional** — split output proportionally across operators
- **Conditional** — apply operator A or B based on a predicate

### Utility
- **Identity** — no-op pass-through
- **WithRate** — apply an operator per-individual with a given probability

## Quick Start

```rust
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::sequential::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let fitness_fn = |args: &[u32; 2]| (args[0] as usize) * (args[0] as usize) - (args[1] as usize);

    let mut ga = EvolutionaryAlgorithm::new(
        Random::new(),
        MaxGenerations::new(100),
        fitness_fn,
        Fill::from_population_size(RandomReset::new()),
        NonZero::new(500).unwrap(),
        rand::rng(),
        Maximize,
    );

    let result = ga.run();
    let best = result.population().best(&fitness_fn, &Maximize);
    println!("Best genome: {:?}, fitness: {:?}", best.genome(), best.fitness(&fitness_fn));
}
```

## Builder Pattern

The GA can also be constructed incrementally with a builder:

```rust
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::sequential::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let fitness_fn = |args: &[u32; 2]| (args[0] as usize) * (args[0] as usize) - (args[1] as usize);

    let mut ga = EvolutionaryAlgorithm::builder(NonZero::new(500).unwrap())
        .initializer(Random::new())
        .termination(MaxGenerations::new(100))
        .fitness(fitness_fn)
        .operators(Fill::from_population_size(RandomReset::new()))
        .rng(rand::rng())
        .comparator(Maximize)
        .build();

    let result = ga.run();
}
```

## Custom Operators

Implement `GeneticOperator` to define your own:

```rust
use evolve::{
    core::{context::Context, offspring::Offspring, state::State},
    operators::GeneticOperator,
};

struct MyOperator;

impl<G, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for MyOperator {
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        // your logic here
        todo!()
    }
}
```

## Parallel Execution

Enable the `parallel` feature to run operators across multiple threads:

```toml
[dependencies]
evolve = { version = "0.3", features = ["parallel"] }
```

Parallel operators distribute work across a thread pool using the `pooled` crate:

```rust
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::parallel::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let mut ga = EvolutionaryAlgorithm::builder(NonZero::new(500).unwrap())
        .initializer(Random::new())
        .termination(MaxGenerations::new(100))
        .fitness(|g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>())
        .operators(Fill::new(RandomReset::new(), 500))
        .rng(rand::rng())
        .comparator(Maximize)
        .build();

    let result = ga.run();
}
```

A `Runtime` is created automatically (defaulting to `available_parallelism` threads) or can be configured via the builder's `.runtime()` method:

```rust
// Use a custom thread pool size
let ga = EvolutionaryAlgorithm::builder(NonZero::new(500).unwrap())
    // ...
    .runtime(pooled::Runtime::new(4))
    .build();
```

## Grammatical Evolution

Evolve programs by mapping integer codon sequences through a grammar. Each individual is a variable-length `Vec<u8>` (or any unsigned integer); codons select productions during derivation, producing a program that is then evaluated for fitness.

### Using the `grammar!` macro (zero-cost, compile-time)

```rust
use evolve::grammar;
use evolve::phenotype::bytecode::{Bytecode, BytecodeBuilder, Instruction};
use evolve::fitness::GeFitness;

grammar! {
    grammar MyGrammar;
    symbol Sym;
    start Expr;

    Expr => [Expr, Expr, BinOp] | [Val];
    BinOp => [Add] | [Sub];
    Val => [X] | [One];
}

impl Instruction for Sym {
    type Value = f64;
    type Input = f64;
    fn execute(&self, stack: &mut Vec<f64>, input: &f64) {
        match self {
            Sym::X => stack.push(*input),
            Sym::One => stack.push(1.0),
            Sym::Add => { let b = stack.pop().unwrap_or(0.0); let a = stack.pop().unwrap_or(0.0); stack.push(a + b); }
            Sym::Sub => { let b = stack.pop().unwrap_or(0.0); let a = stack.pop().unwrap_or(0.0); stack.push(a - b); }
            _ => {}
        }
    }
}
```

### Using the runtime `Grammar<T>` builder

```rust
use evolve::grammar::Grammar;

let grammar = Grammar::builder()
    .rule("expr", &[&["expr", "expr", "op"], &["x"], &["1"]])
    .rule("op", &[&["+"], &["-"]])
    .start("expr")
    .build();
```

### Running GE

```rust
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::{Minimize, GeFitness},
    initialization::RangedRandom,
    operators::sequential::{
        combinator::{Combine, Fill, Pipeline},
        crossover::SinglePoint,
        mutation::RandomReset,
        selection::Tournament,
    },
    termination::MaxGenerations,
};
use std::num::NonZero;

let fitness = GeFitness::<MyGrammar, u8, f64, _, BytecodeBuilder<Sym>, _>::new(
    MyGrammar,
    3,
    |program: &Bytecode<Sym>| {
        // Score the program on test cases
        (program.run(&2.0) - 5.0).abs() // target: f(2) = 5
    },
    f64::MAX,
);

let mut ea = EvolutionaryAlgorithm::new(
    RangedRandom::<u8>::new(10..50),
    MaxGenerations::new(500),
    fitness,
    Fill::from_population_size(Pipeline::new((
        Combine::new((
            Tournament::new(NonZero::new(3).unwrap()),
            Tournament::new(NonZero::new(3).unwrap()),
        )),
        SinglePoint::<u8>::new(),
        RandomReset::<u8>::new(),
    ))),
    NonZero::new(200).unwrap(),
    rand::rng(),
    Minimize,
);

let result = ea.run();
```

### Variable-length operators

GE benefits from variable-length genomes. Use `RangedRandom` for initialization (produces genomes of random length within a range) and the variable-length mutation operators `SegmentDuplication` and `SegmentDeletion` to explore different codon lengths during evolution.

### Custom mappers

The `Mapper` trait allows custom strategies for mapping codon sequences to phenotypes. `StandardMapper` is the default (used automatically by `GeFitness::new()`). Use `GeFitness::with_mapper()` to plug in a custom implementation:

```rust
use evolve::mapper::{Mapper, StandardMapper};
use evolve::grammar::GrammarDef;
use evolve::phenotype::PhenotypeBuilder;
use evolve::codon::Codon;

struct MyMapper;

impl Mapper for MyMapper {
    fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G::Terminal>>(
        &self, grammar: &G, codons: &[C], builder: B,
    ) -> Option<B::Output> {
        // Custom mapping logic
        StandardMapper::new(3).map(grammar, codons, builder)
    }
}

// Use with GeFitness:
// let fitness = GeFitness::with_mapper(grammar, MyMapper, |program| { ... }, penalty);
```

## Collectors

The `run()` method uses a `Standard` collector that records best fitness and timing per generation:

```rust
let result = ea.run();
println!("generations: {}", result.generations());
println!("total time: {:?}", result.total_duration());
println!("best fitness per gen: {:?}", result.best_fitness());
println!("gen 50: {:?}", result.generation(50));
```

Use `run_with` for a different collector:

```rust
use evolve::collector::{Basic, NoOp, Logger};

let result = ea.run_with(Basic);           // just population + generation count
ea.run_with(NoOp);                         // discard everything, returns ()
ea.run_with(Logger::default());            // prints each generation, returns ()
ea.run_with(Logger::with_collector(        // prints + collects
    NonZero::new(10).unwrap(),
    Standard::default(),
));
```

Implement the `Collector` trait for custom data collection:

```rust
use evolve::collector::Collector;
use evolve::core::state::State;

struct DiversityCollector(Vec<usize>);

impl<Fe, C> Collector<[u8; 4], u32, Fe, C> for DiversityCollector {
    type Result = Vec<usize>;

    fn on_generation(&mut self, state: &State<[u8; 4], u32>, _fe: &Fe, _cmp: &C) {
        let unique = state.population().iter()
            .map(|ind| ind.genome())
            .collect::<std::collections::HashSet<_>>()
            .len();
        self.0.push(unique);
    }

    fn finalize(self, _state: State<[u8; 4], u32>) -> Self::Result {
        self.0
    }
}
```

## Experiment Runner

Run multiple independent trials for statistical analysis:

```rust
use evolve::experiment::Experiment;
use evolve::collector::standard::Standard;

let results = Experiment::new(
    || EvolutionaryAlgorithm::new(/* ... */),
    30,
    || Standard::default(),
).run();

for (i, r) in results.iter().enumerate() {
    println!("Run {}: {} gens, {:?}", i, r.generations(), r.total_duration());
}
```

## Serde Support

Enable the `serde` feature to derive `Serialize` and `Deserialize` for core types (`Individual`, `Population`, `State`, `Offspring`) and collector results (`basic::RunResult`, `standard::RunResult`):

```toml
[dependencies]
evolve = { version = "0.3", features = ["serde"] }
```

```rust
use serde_json;

let result = ea.run();
let json = serde_json::to_string(result.population()).unwrap();
println!("{json}");
```

## Contributing

Contributions are welcome! Feel free to open an issue for bug reports, feature requests, or questions. Pull requests are also appreciated.

## AI Disclosure

AI was used only to assist with writing comments, writing tests, writing examples, and as a rubber duck to discuss ideas with. All final decisions and code were written by a human.
