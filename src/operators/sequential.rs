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

use crate::core::{context::Context, offspring::Offspring, state::State};

/// The core trait for all genetic operators (selection, crossover, mutation, combinators, etc.).
///
/// An operator receives the current [`State`] and a [`Context`] and produces an [`Offspring`].
///
/// # Examples
///
/// ```
/// use evolve::core::{context::Context, offspring::Offspring, state::State};
/// use evolve::operators::sequential::GeneticOperator;
///
/// struct MyOperator;
///
/// impl<G, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for MyOperator {
///     fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
///         todo!()
///     }
/// }
/// ```
pub trait GeneticOperator<G, F, Fe, R, C> {
    /// Applies this operator to the current state and returns the resulting offspring.
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F>;
}

#[cfg(test)]
mod test;
