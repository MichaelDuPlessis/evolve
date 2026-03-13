use std::num::NonZero;

use crate::{
    core::{context::Context, population::Population},
    fitness::FitnessEvaluator,
};

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
