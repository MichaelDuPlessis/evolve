use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::operators::GeneticOperator;
use crate::operators::sequential::mutation::creep::CreepMutate;
use rand::{Rng, RngExt, SeedableRng};

use super::GeneCollection;

/// Parallel version of [`Creep`](crate::operators::sequential::mutation::Creep).
///
/// Distributes individuals across pool workers for creep mutation.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct Creep<T> {
    step: T,
}

impl<T> Creep<T> {
    /// Creates a new parallel `Creep` mutation with the given maximum step size.
    pub fn new(step: T) -> Self {
        Self { step }
    }
}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for Creep<T>
where
    G: Clone + AsMut<[T]> + GeneCollection + Send + Sync,
    T: CreepMutate + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let individuals = state.population().as_slice();
        let step = self.step;
        let inputs: vecpool::PoolVec<(u64, usize)> = (0..individuals.len())
            .map(|i| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut genome = individuals[*idx].genome().clone();
            let genes = genome.as_mut();
            let gene_idx = rng.random_range(0..genes.len());
            genes[gene_idx] = genes[gene_idx].creep(step, &mut rng);
            Individual::new(genome)
        });

        let mut population = Population::with_capacity(individuals.len());
        for r in results {
            population.add(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
