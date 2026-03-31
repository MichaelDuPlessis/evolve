//! Genetic operators.
//!
//! - [`sequential`] — single-threaded operators and combinators

use crate::core::{context::Context, offspring::Offspring, state::State};

pub mod parallel;
pub mod sequential;

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

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for &O
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (*self).apply(state, ctx)
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for &mut O
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (**self).apply(state, ctx)
    }
}
