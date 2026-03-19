use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
    random::Randomizable,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// This is a helper trait to get the code to compile since a T may one day implement AsMut<[T]>
trait GeneCollection {}

impl<T> GeneCollection for Vec<T> {}
impl<T> GeneCollection for Box<[T]> {}
impl<T> GeneCollection for [T] {}
impl<T, const N: usize> GeneCollection for [T; N] {}

/// This mutation operator selects a random gene and assigns a new random value to it
#[derive(Debug)]
pub struct RandomReset<T>(PhantomData<T>);

impl<T> RandomReset<T> {
    /// Creates a new `RandomReset` mutation.
    pub fn new() -> Self {
        Self(PhantomData)
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

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for RandomReset<T>
where
    G: Clone + AsMut<[T]> + GeneCollection,
    T: Randomizable<R>,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        // if there is only one no need to allocate a whole Vec
        if state.population().len() == 1 {
            // this is fine, we know there is exactly one element
            let individual = unsafe { state.population().as_slice().get_unchecked(0) };
            Offspring::Single(random_reset_mutate(individual, ctx))
        } else {
            let mut population = Population::with_capacity(state.population().len());

            for individual in state.population() {
                let new_ind = random_reset_mutate(individual, ctx);
                population.add(new_ind);
            }

            Offspring::Multiple(population)
        }
    }
}

macro_rules! impl_random_reset_for_numbers {
    ($($t:ty),*) => {
        $(
            impl<F, R, Fe, C> GeneticOperator<$t, F, Fe, R, C> for RandomReset<$t>
            where
                F: PartialOrd,
                R: Rng,
                Fe: FitnessEvaluator<$t, F>,
            {
                fn apply(&self, state: &State<$t, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<$t, F> {
                    if state.population().len() == 1 {
                        // this is fine, we know there is exactly one element
                        Offspring::Single(Individual::new(
                            <$t>::random(ctx.rng()),
                            ctx.fitness_evaluator(),
                        ))
                    } else {
                        let mut population = Population::with_capacity(state.population().len());

                        for _ in state.population() {
                            let new_ind = Individual::new(<$t>::random(ctx.rng()), ctx.fitness_evaluator());
                            population.add(new_ind);
                        }

                        Offspring::Multiple(population)
                    }
                }
            }
        )*
    };
}

impl_random_reset_for_numbers!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, char
);
