//! # evolve
//!
//! A generic, composable genetic algorithm framework for Rust.
//!
//! `evolve` provides the building blocks to assemble genetic algorithms from reusable,
//! type-safe components. Operators are composed using combinators — chain them into
//! pipelines, weight them probabilistically, or repeat them to fill a population —
//! all with zero-cost abstractions.
//!
//! ## Features
//!
//! - Fully generic over genome type, fitness type, RNG, and fitness comparator
//! - Built-in operators for selection, crossover, and mutation
//! - Composable combinators for structuring the flow of the algorithm
//! - [`Maximize`](fitness::Maximize) and [`Minimize`](fitness::Minimize) fitness comparators out of the box
//! - Closures work as fitness evaluators and comparators via blanket trait impls
//! - No dependencies beyond `rand`
//!
//! ## Quick Start
//!
//! The simplest way to get started is to use [`Random`](initialization::Random) initialization,
//! a [`MaxGenerations`](termination::MaxGenerations) termination condition, and
//! [`Fill`](operators::combinator::Fill) with a mutation operator:
//!
//! ```
//! use evolve::{
//!     algorithm::ga::GeneticAlgorithm,
//!     fitness::Maximize,
//!     initialization::Random,
//!     operators::combinator::Fill,
//!     operators::mutation::RandomReset,
//!     termination::MaxGenerations,
//! };
//! use std::num::NonZero;
//!
//! let mut ga = GeneticAlgorithm::new(
//!     Random::new(),
//!     MaxGenerations::new(100),
//!     |args: &[u32; 2]| args[0] as usize + args[1] as usize,
//!     Fill::from_population_size(RandomReset::new()),
//!     NonZero::new(500).unwrap(),
//!     rand::rng(),
//!     Maximize,
//! );
//!
//! let best = ga.run();
//! println!("Best genome: {:?}, fitness: {:?}", best.genome(), best.fitness());
//! ```
//!
//! ## Composing Operators
//!
//! The real power of `evolve` comes from composing operators using combinators.
//! A typical genetic algorithm pipeline selects parents, crosses them over, and
//! mutates the offspring:
//!
//! ```
//! use evolve::{
//!     algorithm::ga::GeneticAlgorithm,
//!     fitness::Maximize,
//!     initialization::Random,
//!     operators::combinator::{Combine, Fill, Pipeline},
//!     operators::crossover::SinglePoint,
//!     operators::mutation::RandomReset,
//!     operators::selection::TournamentSelection,
//!     termination::MaxGenerations,
//! };
//! use std::num::NonZero;
//!
//! // Select two parents → crossover → mutate, repeated until the population is full
//! let operators = Fill::from_population_size(Pipeline::new((
//!     Combine::new((
//!         TournamentSelection::new(NonZero::new(3).unwrap()),
//!         TournamentSelection::new(NonZero::new(3).unwrap()),
//!     )),
//!     SinglePoint::new(),
//!     RandomReset::new(),
//! )));
//!
//! let mut ga = GeneticAlgorithm::new(
//!     Random::new(),
//!     MaxGenerations::new(200),
//!     |g: &[u8; 8]| g.iter().map(|x| *x as u32).sum::<u32>(),
//!     operators,
//!     NonZero::new(100).unwrap(),
//!     rand::rng(),
//!     Maximize,
//! );
//!
//! let best = ga.run();
//! ```
//!
//! ## Weighted Operator Selection
//!
//! Use [`Weighted`](operators::combinator::Weighted) to probabilistically choose
//! between different operators each time:
//!
//! ```
//! use evolve::operators::combinator::{Fill, Weighted};
//! use evolve::operators::mutation::RandomReset;
//! use evolve::operators::crossover::SinglePoint;
//! use std::num::NonZero;
//!
//! // 75% chance of mutation, 25% chance of crossover
//! let operators = Fill::from_population_size(Weighted::new((
//!     (RandomReset::<u8>::new(), NonZero::new(3u16).unwrap()),
//!     (SinglePoint::<u8>::new(), NonZero::new(1u16).unwrap()),
//! )));
//! ```
//!
//! ## Custom Fitness
//!
//! Any closure `Fn(&G) -> F` works as a [`FitnessEvaluator`](fitness::FitnessEvaluator),
//! and any closure `Fn(&F, &F) -> bool` works as a [`FitnessComparator`](fitness::FitnessComparator):
//!
//! ```
//! use evolve::{
//!     algorithm::ga::GeneticAlgorithm,
//!     initialization::Random,
//!     operators::combinator::Fill,
//!     operators::mutation::RandomReset,
//!     termination::MaxGenerations,
//! };
//! use std::num::NonZero;
//!
//! // Custom comparator: prefer fitness values closer to 100
//! let mut ga = GeneticAlgorithm::new(
//!     Random::new(),
//!     MaxGenerations::new(100),
//!     |g: &[u8; 2]| (g[0] as i32 + g[1] as i32 - 100).abs(),
//!     Fill::from_population_size(RandomReset::new()),
//!     NonZero::new(200).unwrap(),
//!     rand::rng(),
//!     |a: &i32, b: &i32| a < b, // lower distance is better
//! );
//!
//! let best = ga.run();
//! ```
//!
//! ## Custom Operators
//!
//! Implement [`GeneticOperator`](operators::GeneticOperator) to define your own:
//!
//! ```
//! use evolve::{
//!     core::{context::Context, offspring::Offspring, state::State},
//!     operators::GeneticOperator,
//! };
//!
//! struct MyOperator;
//!
//! impl<G, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for MyOperator {
//!     fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
//!         todo!()
//!     }
//! }
//! ```

pub mod algorithm;
pub mod core;
pub mod fitness;
pub mod initialization;
pub mod operators;
pub mod random;
pub mod termination;
