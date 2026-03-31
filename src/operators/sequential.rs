//! Genetic operators and combinators.
//!
//! - [`selection`] — operators that select individuals from the population
//! - [`crossover`] — operators that recombine genomes
//! - [`mutation`] — operators that introduce random changes
//! - [`combinator`] — composable wrappers that structure operator flow

pub mod combinator;
pub mod crossover;
pub mod mutation;
pub mod selection;

#[cfg(test)]
mod test;
