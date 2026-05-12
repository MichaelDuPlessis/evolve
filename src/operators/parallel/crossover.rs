use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{GeneticOperator, common::single_point_crossover},
};
use rand::{Rng, RngExt, SeedableRng};
use std::marker::PhantomData;

/// Parallel version of [`SinglePoint`](crate::operators::sequential::crossover::SinglePoint).
///
/// Distributes pairs of individuals across pool workers for crossover.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct SinglePoint<T> {
    _marker: PhantomData<T>,
}

impl<T> SinglePoint<T> {
    /// Creates a new parallel `SinglePoint` crossover operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for SinglePoint<T> {
    fn default() -> Self {
        Self::new()
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
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[T; N], F> {
        let individuals = state.population().as_slice();
        let inputs: Vec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = single_point_crossover(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}
