use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{common::single_point_crossover, GeneticOperator},
};
use rand::{Rng, RngExt, SeedableRng};
use std::{marker::PhantomData, num::NonZero};

/// Parallel version of [`SinglePoint`](crate::operators::sequential::crossover::SinglePoint).
///
/// Distributes pairs of individuals across threads for crossover.
/// Each thread gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct SinglePoint<T> {
    num_threads: usize,
    _marker: PhantomData<T>,
}

impl<T> SinglePoint<T> {
    /// Creates a new parallel `SinglePoint` crossover operator.
    pub fn new(num_threads: NonZero<usize>) -> Self {
        Self {
            num_threads: num_threads.get(),
            _marker: PhantomData,
        }
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for SinglePoint<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<[T; N], F>: Sync,
{
    fn apply(
        &self,
        state: &State<[T; N], F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<[T; N], F> {
        let individuals = state.population().as_slice();
        let pair_count = individuals.len() / 2;
        let chunk_size = (pair_count / self.num_threads).max(1) * 2;

        let seeds: Vec<u64> = (0..self.num_threads)
            .map(|_| ctx.rng().random())
            .collect();

        let mut population = Population::with_capacity(pair_count * 2);

        std::thread::scope(|s| {
            let handles: Vec<_> = individuals
                .chunks(chunk_size)
                .zip(seeds)
                .map(|(chunk, seed)| {
                    s.spawn(move || -> Population<[T; N], F> {
                        let mut rng = R::seed_from_u64(seed);
                        let mut pop = Population::with_capacity(chunk.len());
                        for pair in chunk.chunks_exact(2) {
                            let (c1, c2) = single_point_crossover(
                                pair[0].genome(),
                                pair[1].genome(),
                                &mut rng,
                            );
                            pop.add(Individual::new(c1));
                            pop.add(Individual::new(c2));
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
