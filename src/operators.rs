use crate::{core::population::Population, fitness::FitnessEvaluation};

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F> {
    fn apply(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F>;
}
