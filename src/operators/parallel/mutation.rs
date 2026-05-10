use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{common::random_reset_mutate, GeneticOperator},
    random::Randomizable,
};
use rand::{Rng, RngExt, SeedableRng};
use std::marker::PhantomData;

/// Parallel version of [`RandomReset`](crate::operators::sequential::mutation::RandomReset).
///
/// Distributes individuals across pool workers for mutation.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct RandomReset<T> {
    _marker: PhantomData<T>,
}

impl<T> RandomReset<T> {
    /// Creates a new parallel `RandomReset` mutation operator.
    pub fn new() -> Self {
        Self {
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
        let inputs: Vec<(u64, usize)> = (0..individuals.len())
            .map(|i| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            Individual::new(random_reset_mutate(individuals[*idx].genome(), &mut rng))
        });

        let mut population = Population::with_capacity(individuals.len());
        for r in results {
            population.add(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
