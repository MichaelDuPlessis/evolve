use crate::{
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluator,
    random::Randomizable,
};
use rand::Rng;
use std::num::NonZero;

/// Trait to initialize the population
pub trait Initializer<G, F, Fe, R>
where
    Fe: FitnessEvaluator<G, F>,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R>,
    ) -> Population<G, F>;
}

/// Create a population by randomly creating their genomes.
pub struct Random;

impl Random {
    /// Create a new `Random` intializer.
    pub fn new() -> Self {
        Self
    }
}

impl<G, F, Fe, R> Initializer<G, F, Fe, R> for Random
where
    Fe: FitnessEvaluator<G, F>,
    G: Randomizable<R>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R>,
    ) -> Population<G, F> {
        (0..population_size.get())
            .map(|_| Individual::new(G::random(ctx.rng()), ctx.fitness_evaluator()))
            .collect()
    }
}
