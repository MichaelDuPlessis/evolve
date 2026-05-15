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
/// An operator receives the current [`State`] (population + generation) and a
/// [`Context`] (fitness evaluator + RNG + comparator) and produces [`Offspring`].
///
/// # Two methods
///
/// - [`apply`](Self::apply) — receives a borrowed state. Use this when the caller
///   retains ownership of the population (e.g., [`Fill`](crate::operators::sequential::combinator::Fill)
///   calls the same operator repeatedly against the same population).
///
/// - [`transform`](Self::transform) — receives an owned state. Used by
///   [`Pipeline`](crate::operators::sequential::combinator::Pipeline) to pass
///   data between chained operators. Operators that can modify individuals in place
///   (e.g., mutation) should override this to avoid unnecessary cloning.
///
/// Most operators only need to implement [`apply`](Self::apply). The default
/// [`transform`](Self::transform) delegates to it.
///
/// # Examples
///
/// A simple operator that only implements `apply`:
///
/// ```
/// use evolve::core::{context::Context, offspring::Offspring, state::State};
/// use evolve::core::individual::Individual;
/// use evolve::operators::GeneticOperator;
/// use evolve::fitness::FitnessEvaluator;
/// use rand::{Rng, RngExt};
///
/// /// Replaces every individual's genome with a random value.
/// struct Randomize;
///
/// impl<F, R: Rng, Fe: FitnessEvaluator<u32, F>, C> GeneticOperator<u32, F, Fe, R, C> for Randomize {
///     fn apply(&self, state: &State<u32, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<u32, F> {
///         let ind = Individual::new(ctx.rng().random::<u32>());
///         Offspring::Single(ind)
///     }
/// }
/// ```
///
/// An operator that overrides `transform` for in-place mutation:
///
/// ```
/// use evolve::core::{context::Context, offspring::Offspring, population::Population, state::State};
/// use evolve::core::individual::Individual;
/// use evolve::operators::GeneticOperator;
/// use evolve::fitness::FitnessEvaluator;
/// use rand::Rng;
///
/// /// Doubles every genome value in place.
/// struct DoubleGenome;
///
/// impl<F, R: Rng, Fe: FitnessEvaluator<u32, F>, C> GeneticOperator<u32, F, Fe, R, C> for DoubleGenome {
///     fn apply(&self, state: &State<u32, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<u32, F> {
///         // Fallback: clone and modify
///         let pop: Population<u32, F> = state.population().iter()
///             .map(|ind| Individual::new(ind.genome() * 2))
///             .collect();
///         Offspring::Multiple(pop)
///     }
///
///     fn transform(&self, state: State<u32, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<u32, F> {
///         // Optimized: modify in place without cloning
///         let pop: Population<u32, F> = state.into_population().into_iter()
///             .map(|ind| ind.mutate_genome(|g| *g *= 2))
///             .collect();
///         Offspring::Multiple(pop)
///     }
/// }
/// ```
pub trait GeneticOperator<G, F, Fe, R, C> {
    /// Applies this operator to a borrowed state and returns the resulting offspring.
    ///
    /// The state is immutable — the operator reads the population but cannot
    /// modify it. This is the primary method to implement for any operator.
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F>;

    /// Applies this operator to an owned state, allowing in-place modification.
    ///
    /// Called by [`Pipeline`](crate::operators::sequential::combinator::Pipeline)
    /// when chaining operators. Since the state is owned, operators can
    /// destructure it and modify individuals directly without cloning.
    ///
    /// The default implementation borrows the owned state and delegates to
    /// [`apply`](Self::apply). Override this in operators (like mutation) that
    /// can transform individuals in place for better performance.
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
