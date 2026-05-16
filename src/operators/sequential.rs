//! Genetic operators and combinators.
//!
//! - [`selection`] — operators that select individuals from the population
//! - [`crossover`] — operators that recombine genomes
//! - [`mutation`] — operators that introduce random changes
//! - [`combinator`] — composable wrappers that structure operator flow

pub mod combinator;
pub mod crossover;
/// Identity (no-op) operator.
pub mod identity;
pub mod mutation;
pub mod selection;
pub mod with_rate;

#[cfg(test)]
mod test;
