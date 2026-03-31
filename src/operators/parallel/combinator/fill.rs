use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::{GeneticOperator, sequential::combinator::Fill as SequentialFill},
};
use rand::{Rng, RngExt, SeedableRng};
use std::num::NonZero;

/// Parallel version of [`Fill`](crate::operators::sequential::combinator::Fill).
///
/// Splits the target size across multiple threads, each running a sequential
/// [`Fill`](crate::operators::sequential::combinator::Fill) with its own RNG.
/// Results are merged into a single population.
///
/// Requires the operator to be `Clone + Send + Sync` and the RNG to be `SeedableRng`.
#[derive(Debug, Clone)]
pub struct Fill<O> {
    operator: O,
    target_size: usize,
    num_threads: usize,
}

impl<O> Fill<O> {
    /// Creates a new parallel `Fill`.
    pub fn new(operator: O, target_size: usize, num_threads: NonZero<usize>) -> Self {
        Self {
            operator,
            target_size,
            num_threads: num_threads.get(),
        }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Fill<O>
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
        let chunk_size = self.target_size / self.num_threads;
        let extra = self.target_size % self.num_threads;

        let seeds: Vec<u64> = (0..self.num_threads).map(|_| ctx.rng().random()).collect();
        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();

        let mut population = Population::with_capacity(self.target_size);

        std::thread::scope(|s| {
            let handles: Vec<_> = seeds
                .into_iter()
                .enumerate()
                .map(|(i, seed)| {
                    let size = if i < extra {
                        chunk_size + 1
                    } else {
                        chunk_size
                    };

                    let fill = SequentialFill::from_fixed_size(&self.operator, size);
                    s.spawn(move || {
                        let mut rng = R::seed_from_u64(seed);
                        let mut thread_ctx = Context::new(fe, &mut rng, comp);
                        fill.apply(state, &mut thread_ctx)
                    })
                })
                .collect();

            for handle in handles {
                population.add_offspring(handle.join().unwrap());
            }
        });

        Offspring::Multiple(population)
    }
}
