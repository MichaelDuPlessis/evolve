use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt, SeedableRng};
use std::marker::PhantomData;

use super::GeneCollection;

/// Parallel version of [`Scramble`](crate::operators::sequential::mutation::Scramble).
///
/// Distributes individuals across pool workers for scramble mutation.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct Scramble<T> {
    _marker: PhantomData<T>,
}

impl<T> Scramble<T> {
    /// Creates a new parallel `Scramble` mutation operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Scramble<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for Scramble<T>
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
                let mut a = rng.random_range(0..genes.len());
                let mut b = rng.random_range(0..genes.len());
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                for i in (a + 1..=b).rev() {
                    let j = rng.random_range(a..=i);
                    genes.swap(i, j);
                }
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
