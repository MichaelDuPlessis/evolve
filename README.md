# evolve

A generic, composable genetic algorithm framework for Rust.

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
    algorithm::ga::GeneticAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::sequential::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let fitness_fn = |args: &[u32; 2]| (args[0] as usize) * (args[0] as usize) - (args[1] as usize);

    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(100),
        fitness_fn,
        Fill::from_population_size(RandomReset::new()),
        NonZero::new(500).unwrap(),
        rand::rng(),
        Maximize,
    );

    let result = ga.run();
    let best = result.population.best(&fitness_fn, &Maximize);
    println!("Best genome: {:?}, fitness: {:?}", best.genome(), best.fitness(&fitness_fn));
}
```

## Builder Pattern

The GA can also be constructed incrementally with a builder:

```rust
use evolve::{
    algorithm::ga::GeneticAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::sequential::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let fitness_fn = |args: &[u32; 2]| (args[0] as usize) * (args[0] as usize) - (args[1] as usize);

    let mut ga = GeneticAlgorithm::builder(NonZero::new(500).unwrap())
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
evolve = { version = "0.1.0", features = ["parallel"] }
```

Parallel operators distribute work across a thread pool using the `pooled` crate:

```rust
use evolve::{
    algorithm::ga::GeneticAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::parallel::combinator::Fill,
    operators::sequential::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let mut ga = GeneticAlgorithm::builder(NonZero::new(500).unwrap())
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
let ga = GeneticAlgorithm::builder(NonZero::new(500).unwrap())
    // ...
    .runtime(pooled::Runtime::new(4))
    .build();
```

## Contributing

Contributions are welcome! Feel free to open an issue for bug reports, feature requests, or questions. Pull requests are also appreciated.

## AI Disclosure

AI was used only to assist with writing comments, writing tests, writing examples, and as a rubber duck to discuss ideas with. All final decisions and code were written by a human.
