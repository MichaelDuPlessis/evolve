# evolve

A generic, composable genetic algorithm framework for Rust.

> **Note:** This library is in rapid development. While breaking changes will be avoided where possible, there is no guarantee of API stability until 1.0.

`evolve` provides the building blocks to assemble genetic algorithms from reusable, type-safe components. Operators are composed using combinators — chain them into pipelines, weight them probabilistically, or repeat them to fill a population — all with zero-cost abstractions.

## Features

- Fully generic over genome type, fitness type, RNG, and fitness comparator
- Built-in operators for selection, crossover, and mutation
- Composable combinators for structuring the flow of the algorithm
- `Maximize` and `Minimize` fitness comparators out of the box
- Closures work as fitness evaluators and comparators via blanket trait impls
- No dependencies beyond `rand` (optional `pooled` for parallel execution)

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
evolve = { version = "0.2.0", features = ["parallel"] }
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
        selection::TournamentSelection,
    },
    termination::MaxGenerations,
};
use std::num::NonZero;

let fitness = GeFitness::<MyGrammar, u8, f64, _, BytecodeBuilder<Sym>>::new(
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
            TournamentSelection::new(NonZero::new(3).unwrap()),
            TournamentSelection::new(NonZero::new(3).unwrap()),
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

## Contributing

Contributions are welcome! Feel free to open an issue for bug reports, feature requests, or questions. Pull requests are also appreciated.

## AI Disclosure

AI was used only to assist with writing comments, writing tests, writing examples, and as a rubber duck to discuss ideas with. All final decisions and code were written by a human.
