use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// Mutation operator that deletes a random segment from a genome.
///
/// The segment length is bounded by `max_segment_fraction` of the genome length
/// and will never reduce the genome below `min_genome_len`.
#[derive(Debug, Clone, Copy)]
pub struct SegmentDeletion<T> {
    max_segment_fraction: f64,
    min_genome_len: usize,
    _marker: PhantomData<T>,
}

impl<T> SegmentDeletion<T> {
    /// Creates a new `SegmentDeletion` operator.
    ///
    /// - `max_segment_fraction` — maximum fraction of the genome to delete (0.0..1.0)
    /// - `min_genome_len` — deletion is skipped if it would reduce below this length
    pub fn new(max_segment_fraction: f64, min_genome_len: usize) -> Self {
        Self {
            max_segment_fraction,
            min_genome_len,
            _marker: PhantomData,
        }
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for SegmentDeletion<T>
where
    T: Clone,
    R: Rng,
    Fe: FitnessEvaluator<Vec<T>, F>,
{
    fn apply(&self, state: &State<Vec<T>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<T>, F> {
        let mut population = Population::with_capacity(state.population().len());

        for individual in state.population() {
            let genome = individual.genome();
            let len = genome.len();
            let max_by_fraction = (len as f64 * self.max_segment_fraction).floor() as usize;
            let max_by_min_len = len.saturating_sub(self.min_genome_len);
            let max_deletable = max_by_fraction.min(max_by_min_len);

            if max_deletable < 1 {
                population.add(Individual::new(genome.clone()));
            } else {
                let seg_len = ctx.rng().random_range(1..=max_deletable);
                let start = ctx.rng().random_range(0..=(len - seg_len));
                let mut new_genome: vecpool::PoolVec<T> = vecpool::with_capacity(len - seg_len);
                new_genome.extend_from_slice(&genome[..start]);
                new_genome.extend_from_slice(&genome[start + seg_len..]);
                population.add(Individual::new(new_genome.into_vec()));
            }
        }

        Offspring::Multiple(population)
    }

    fn transform(
        &self,
        state: State<Vec<T>, F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<Vec<T>, F> {
        let population: Population<Vec<T>, F> = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let len = genome.len();
                    let max_by_fraction = (len as f64 * self.max_segment_fraction).floor() as usize;
                    let max_by_min_len = len.saturating_sub(self.min_genome_len);
                    let max_deletable = max_by_fraction.min(max_by_min_len);
                    if max_deletable < 1 {
                        return;
                    }
                    let seg_len = ctx.rng().random_range(1..=max_deletable);
                    let start = ctx.rng().random_range(0..=(len - seg_len));
                    genome.drain(start..start + seg_len);
                })
            })
            .collect();

        Offspring::Multiple(population)
    }
}
