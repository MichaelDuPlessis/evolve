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
    min: Option<T>,
    max: Option<T>,
}

impl<T> Creep<T> {
    /// Creates a new parallel `Creep` mutation with the given maximum step size.
    pub fn new(step: T) -> Self {
        Self {
            step,
            min: None,
            max: None,
        }
    }

    /// Sets the lower bound. Genes are clamped to this minimum after mutation.
    pub fn min(mut self, min: T) -> Self {
        self.min = Some(min);
        self
    }

    /// Sets the upper bound. Genes are clamped to this maximum after mutation.
    pub fn max(mut self, max: T) -> Self {
        self.max = Some(max);
        self
    }
}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for Creep<T>
where
    G: Clone + AsMut<[T]> + GeneCollection + Send + Sync,
    T: CreepMutate + Ord + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let individuals = state.population().as_slice();
        let step = self.step;
        let min = self.min;
        let max = self.max;
        let inputs: vecpool::PoolVec<(u64, usize)> = (0..individuals.len())
            .map(|i| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut genome = individuals[*idx].genome().clone();
            let genes = genome.as_mut();
            let gene_idx = rng.random_range(0..genes.len());
            genes[gene_idx] = genes[gene_idx].creep(step, &mut rng);
            if let Some(lo) = min {
                genes[gene_idx] = std::cmp::max(genes[gene_idx], lo);
            }
            if let Some(hi) = max {
                genes[gene_idx] = std::cmp::min(genes[gene_idx], hi);
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
