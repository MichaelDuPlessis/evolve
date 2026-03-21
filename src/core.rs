//! Core types that make up the genetic algorithm.
//!
//! - [`Individual`](individual::Individual) — a genome paired with its fitness value
//! - [`Population`](population::Population) — a collection of individuals
//! - [`Offspring`](offspring::Offspring) — the result of applying a genetic operator
//! - [`State`](state::State) — the current generation and population
//! - [`Context`](context::Context) — shared resources passed to operators
//! - [`RunResult`](run_result::RunResult) — the output of running an algorithm

pub mod context;
pub mod individual;
pub mod offspring;
pub mod population;
pub mod run_result;
pub mod state;

#[cfg(test)]
mod test;
