use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{common::random_reset_mutate, GeneticOperator},
    random::Randomizable,
};
use rand::{Rng, RngExt, SeedableRng};
use std::{marker::PhantomData, num::NonZero};

/// Parallel version of [`RandomReset`](crate::operators::sequential::mutation::RandomReset).
///
/// Distributes individuals across threads for mutation.
/// Each thread gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct RandomReset<T> {
    num_threads: usize,
    _marker: PhantomData<T>,
}

impl<T> RandomReset<T> {
    /// Creates a new parallel `RandomReset` mutation operator.
    pub fn new(num_threads: NonZero<usize>) -> Self {
        Self {
            num_threads: num_threads.get(),
            _marker: PhantomData,
        }
    }
}

/// Helper trait — see sequential mutation module.
trait GeneCollection {}
impl<T> GeneCollection for Vec<T> {}
impl<T> GeneCollection for Box<[T]> {}
impl<T> GeneCollection for [T] {}
impl<T, const N: usize> GeneCollection for [T; N] {}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for RandomReset<T>
where
    G: Clone + AsMut<[T]> + GeneCollection + Send + Sync,
    T: Randomizable<R> + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let individuals = state.population().as_slice();
        let chunk_size = (individuals.len() / self.num_threads).max(1);

        let seeds: Vec<u64> = (0..self.num_threads)
            .map(|_| ctx.rng().random())
            .collect();

        let mut population = Population::with_capacity(individuals.len());

        std::thread::scope(|s| {
            let handles: Vec<_> = individuals
                .chunks(chunk_size)
                .zip(seeds)
                .map(|(chunk, seed)| {
                    s.spawn(move || -> Population<G, F> {
                        let mut rng = R::seed_from_u64(seed);
                        let mut pop = Population::with_capacity(chunk.len());
                        for ind in chunk {
                            pop.add(Individual::new(random_reset_mutate(ind.genome(), &mut rng)));
                        }
                        pop
                    })
                })
                .collect();

            for handle in handles {
                population.merge(handle.join().unwrap());
            }
        });

        Offspring::Multiple(population)
    }
}
