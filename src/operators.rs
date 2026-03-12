use crate::{
    core::{context::Context, population::Population},
    fitness::FitnessEvaluation,
};

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F, Fe>
where
    Fe: FitnessEvaluation<G, F>,
{
    fn generate(&self, ctx: &Context<G, F, Fe>) -> Population<G, F>;
}
