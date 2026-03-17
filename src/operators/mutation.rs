use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
        offspring::Offpring,
        population::Population,
    },
    fitness::FitnessEvaluator,
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

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for RandomReset<T>
where
    G: Clone + AsMut<[T]>,
    T: Randomizable<R>,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offpring<G, F> {
        // if there is only one no need to allocate a whole Vec
        if state.population().len() == 1 {
            // this is fine, we know there is exactly one element
            let individual = unsafe { state.population().as_slice().get_unchecked(0) };
            Offpring::Single(random_reset_mutate(individual, ctx))
        } else {
            let mut population = Population::with_capacity(state.population().len());

            for individual in state.population() {
                let new_ind = random_reset_mutate(individual, ctx);
                population.add(new_ind);
            }

            Offpring::Multiple(population)
        }
    }
}

/// Mutates a single genome using the random reset pattern
fn random_reset_mutate<G, F, R, Fe, T, C>(
    individual: &Individual<G, F>,
    ctx: &mut Context<Fe, R, C>,
) -> Individual<G, F>
where
    G: Clone + AsMut<[T]>,
    T: Randomizable<R>,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    let mut new_genome = individual.genome().clone();

    // Convert genome to a mutable slice of genes
    let genes = new_genome.as_mut();

    // Pick a random gene index
    let gene_index = ctx.rng().random_range(0..genes.len());

    // Replace it with a new random value
    genes[gene_index] = T::random(ctx.rng());

    Individual::new(new_genome, ctx.fitness_evaluator())
}
