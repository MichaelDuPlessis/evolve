use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::GeneticOperator,
};
use rand::{Rng, RngExt, SeedableRng};
use std::marker::PhantomData;

use super::GeneCollection;

/// Parallel version of [`Swap`](crate::operators::sequential::mutation::Swap).
///
/// Distributes individuals across pool workers for swap mutation.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct Swap<T> {
    _marker: PhantomData<T>,
}

impl<T> Swap<T> {
    /// Creates a new parallel `Swap` mutation operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Swap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for Swap<T>
where
    G: Clone + AsMut<[T]> + GeneCollection + Send + Sync,
    T: Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = (0..individuals.len())
            .map(|i| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut genome = individuals[*idx].genome().clone();
            let genes = genome.as_mut();
            if genes.len() >= 2 {
                let a = rng.random_range(0..genes.len());
                let b = rng.random_range(0..genes.len());
                genes.swap(a, b);
            }
            Individual::new(genome)
        });

        let mut population = Population::with_capacity(individuals.len());
        for r in results {
            population.add(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
