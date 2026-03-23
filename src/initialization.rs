//! Population initialization strategies.
//!
//! Defines the [`Initializer`] trait and provides [`Random`], which generates
//! a population of random genomes.

use crate::{
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluator,
    random::Randomizable,
};
use rand::Rng;
use std::num::NonZero;

/// Creates the initial population for the algorithm.
///
/// # Examples
///
/// ```
/// use evolve::core::context::Context;
/// use evolve::fitness::Maximize;
/// use evolve::initialization::{Initializer, Random};
/// use std::num::NonZero;
///
/// let mut rng = rand::rng();
/// let fitness_fn = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
/// let mut ctx = Context::new(&fitness_fn, &mut rng, &Maximize);
///
/// let pop = Random::new().initialize(NonZero::new(100).unwrap(), &mut ctx);
/// assert_eq!(pop.len(), 100);
/// ```
pub trait Initializer<G, F, Fe, R, C>
where
    Fe: FitnessEvaluator<G, F>,
{
    /// Creates the initial population of the given size.
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F>;
}

/// An [`Initializer`] that creates a population of random genomes.
///
/// Requires the genome type to implement [`Randomizable`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Random;

impl Random {
    /// Creates a new `Random` initializer.
    pub fn new() -> Self {
        Self
    }
}

impl<G, F, Fe, R, C> Initializer<G, F, Fe, R, C> for Random
where
    Fe: FitnessEvaluator<G, F>,
    G: Randomizable<R>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F> {
        (0..population_size.get())
            .map(|_| Individual::new(G::random(ctx.rng()), ctx.fitness_evaluator()))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::context::Context;
    use crate::fitness::Maximize;

    fn id(g: &[u8; 2]) -> u16 {
        g[0] as u16 + g[1] as u16
    }

    #[test]
    fn random_initializer_creates_correct_size() {
        let mut rng = rand::rng();
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        let pop = Random::new().initialize(NonZero::new(10).unwrap(), &mut ctx);
        assert_eq!(pop.len(), 10);
    }

    #[test]
    fn random_initializer_evaluates_fitness() {
        let mut rng = rand::rng();
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        let pop = Random::new().initialize(NonZero::new(5).unwrap(), &mut ctx);
        for ind in &pop {
            assert_eq!(
                *ind.fitness(),
                ind.genome()[0] as u16 + ind.genome()[1] as u16
            );
        }
    }
}
