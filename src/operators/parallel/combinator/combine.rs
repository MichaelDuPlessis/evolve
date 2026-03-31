use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};
use rand::{Rng, RngExt, SeedableRng};
use std::num::NonZero;

/// Parallel version of [`Combine`](crate::operators::sequential::combinator::Combine).
///
/// Distributes a slice of operators across threads, each running on the same
/// input with its own RNG. Results are merged into one population.
///
/// Only available for homogeneous operator slices `[O]`. For heterogeneous
/// operator tuples, use the sequential [`Combine`](crate::operators::sequential::combinator::Combine).
#[derive(Debug, Clone)]
pub struct Combine<O: ?Sized> {
    num_threads: usize,
    operators: O,
}

impl<O> Combine<O> {
    /// Creates a new parallel `Combine`.
    pub fn new(operators: O, num_threads: NonZero<usize>) -> Self {
        Self {
            num_threads: num_threads.get(),
            operators,
        }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Combine<[O]>
where
    O: GeneticOperator<G, F, Fe, R, C> + Sync,
    G: Send,
    F: Send,
    Fe: Sync,
    R: Rng + SeedableRng,
    C: Sync,
    State<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let chunk_size = (self.operators.len() / self.num_threads).max(1);

        let seeds: Vec<u64> = (0..self.num_threads)
            .map(|_| ctx.rng().random())
            .collect();
        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();

        let mut population = Population::new();

        std::thread::scope(|s| {
            let handles: Vec<_> = self
                .operators
                .chunks(chunk_size)
                .zip(seeds)
                .map(|(chunk, seed)| {
                    s.spawn(move || -> Population<G, F> {
                        let mut rng = R::seed_from_u64(seed);
                        let mut thread_ctx = Context::new(fe, &mut rng, comp);
                        let mut pop = Population::new();
                        for op in chunk {
                            pop.add_offspring(op.apply(state, &mut thread_ctx));
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
