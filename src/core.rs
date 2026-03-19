//! Core types that make up the genetic algorithm.
//!
//! This module contains the fundamental building blocks:
//! - [`Individual`](individual::Individual) — a genome paired with its fitness value
//! - [`Population`](population::Population) — a collection of individuals
//! - [`Offspring`](offspring::Offspring) — the result of applying a genetic operator
//! - [`State`](state::State) — the current generation and population
//! - [`Context`](context::Context) — shared resources (fitness evaluator, RNG, comparator) passed to operators

//! Core types that make up the genetic algorithm.
//!
//! - [`Individual`](individual::Individual) — a genome paired with its fitness value
//! - [`Population`](population::Population) — a collection of individuals
//! - [`Offspring`](offspring::Offspring) — the result of applying a genetic operator
//! - [`State`](state::State) — the current generation and population
//! - [`Context`](context::Context) — shared resources passed to operators

pub mod context;
pub mod individual;
pub mod offspring;
pub mod population;
pub mod state;

#[cfg(test)]
mod test;
