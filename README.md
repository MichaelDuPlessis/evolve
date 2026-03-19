# evolve

A generic, composable genetic algorithm framework for Rust.

`evolve` provides the building blocks to assemble genetic algorithms from reusable, type-safe components. Operators are composed using combinators — chain them into pipelines, weight them probabilistically, or repeat them to fill a population — all with zero-cost abstractions.

## Features

- Fully generic over genome type, fitness type, RNG, and fitness comparator
- Built-in operators for selection, crossover, and mutation
- Composable combinators for structuring the flow of the algorithm
- `Maximize` and `Minimize` fitness comparators out of the box
- Closures work as fitness evaluators and comparators via blanket trait impls
- No dependencies beyond `rand`

## Quick Start

```rust
use evolve::{
    algorithm::ga::GeneticAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::combinator::Fill,
    operators::mutation::RandomReset,
    termination::MaxGenerations,
};
use std::num::NonZero;

fn main() {
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(100),
        |args: &[u32; 2]| (args[0] as usize) * (args[0] as usize) - (args[1] as usize),
        Fill::from_population_size(RandomReset::new()),
        NonZero::new(500).unwrap(),
        rand::rng(),
        Maximize,
    );

    let best = ga.run();
    println!("Best genome: {:?}, fitness: {:?}", best.genome(), best.fitness());
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

## AI Disclosure

AI was used only to assist with writing comments, writing tests, writing examples, and as a rubber duck to discuss ideas with. All final decisions and code were written by a human.
