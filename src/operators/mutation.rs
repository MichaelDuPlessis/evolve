use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
        population::Population,
    },
    fitness::FitnessEvaluation,
    operators::GeneticOperator,
    random::Randomizable,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// This mutation operator selects a random gene and assigns a new random value to it
pub struct RandomReset<T>(PhantomData<T>);

impl<T> RandomReset<T> {
    /// Creates a new `RandomReset` mutation.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<G, F, R, Fe, T> GeneticOperator<G, F, Fe, R> for RandomReset<T>
where
    G: Clone + AsMut<[T]>,
    T: Randomizable<R>,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluation<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R>) -> Population<G, F> {
        let mut offspring = Population::with_capacity(state.population().len());

        for individual in state.population() {
            let mut new_genome = individual.genome.clone();

            // Convert genome to a mutable slice of genes
            let genes = new_genome.as_mut();

            // Pick a random gene index
            let gene_index = ctx.rng().random_range(0..genes.len());

            // Replace it with a new random value
            genes[gene_index] = T::random(ctx.rng());

            let new_ind = Individual::new(new_genome);
            offspring.add(new_ind);
        }

        offspring
    }
}
