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

/// Mutation operator that duplicates a random segment within a genome.
///
/// The duplicated segment is inserted immediately after its original position.
/// If the resulting genome would exceed `max_genome_len`, the individual is left unchanged.
#[derive(Debug, Clone, Copy)]
pub struct SegmentDuplication<T> {
    max_segment_fraction: f64,
    max_genome_len: usize,
    _marker: PhantomData<T>,
}

impl<T> SegmentDuplication<T> {
    /// Creates a new `SegmentDuplication` operator.
    ///
    /// - `max_segment_fraction` — maximum fraction of the genome to duplicate (0.0..1.0)
    /// - `max_genome_len` — duplication is skipped if it would exceed this length
    pub fn new(max_segment_fraction: f64, max_genome_len: usize) -> Self {
        Self {
            max_segment_fraction,
            max_genome_len,
            _marker: PhantomData,
        }
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for SegmentDuplication<T>
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

            if len == 0 {
                population.add(Individual::new(genome.clone()));
                continue;
            }

            let max_seg = ((len as f64 * self.max_segment_fraction).floor() as usize).max(1);
            let seg_len = ctx.rng().random_range(1..=max_seg);

            if len + seg_len > self.max_genome_len {
                population.add(Individual::new(genome.clone()));
            } else {
                let start = ctx.rng().random_range(0..=(len - seg_len));
                let mut new_genome: vecpool::PoolVec<T> = vecpool::with_capacity(len + seg_len);
                new_genome.extend_from_slice(&genome[..start + seg_len]);
                new_genome.extend_from_slice(&genome[start..start + seg_len]);
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
                    if len == 0 {
                        return;
                    }
                    let max_seg =
                        ((len as f64 * self.max_segment_fraction).floor() as usize).max(1);
                    let seg_len = ctx.rng().random_range(1..=max_seg);
                    if len + seg_len > self.max_genome_len {
                        return;
                    }
                    let start = ctx.rng().random_range(0..=(len - seg_len));
                    let mut segment: vecpool::PoolVec<T> = vecpool::with_capacity(seg_len);
                    segment.extend_from_slice(&genome[start..start + seg_len]);
                    genome.splice(start + seg_len..start + seg_len, segment.into_vec());
                })
            })
            .collect();

        Offspring::Multiple(population)
    }
}
