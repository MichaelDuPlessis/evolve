//! Genetic operators.
//!
//! - [`sequential`] — single-threaded operators and combinators

use crate::core::{context::Context, offspring::Offspring, state::State};

pub(crate) mod common;
#[cfg(feature = "parallel")]
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
/// use evolve::operators::GeneticOperator;
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

    /// Applies this operator to an owned state, allowing in-place modification.
    ///
    /// The default implementation borrows the state and delegates to [`apply`](Self::apply).
    /// Operators that can work in-place (e.g., mutation) may override this to
    /// avoid cloning genomes.
    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        self.apply(&state, ctx)
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for &O
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (*self).apply(state, ctx)
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (*self).transform(state, ctx)
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for &mut O
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (**self).apply(state, ctx)
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        (**self).transform(state, ctx)
    }
}
