use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};
use rand::{Rng, RngExt, SeedableRng};
use std::num::NonZero;

/// Parallel version of [`Repeat`](crate::operators::sequential::combinator::Repeat).
///
/// Distributes `n` repetitions across threads, each with its own RNG.
/// Results are merged into one population.
#[derive(Debug, Clone)]
pub struct Repeat<O> {
    operator: O,
    n: usize,
    num_threads: usize,
}

impl<O> Repeat<O> {
    /// Creates a new parallel `Repeat`.
    pub fn new(operator: O, n: usize, num_threads: NonZero<usize>) -> Self {
        Self {
            operator,
            n,
            num_threads: num_threads.get(),
        }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Repeat<O>
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
        let chunk_size = self.n / self.num_threads;
        let extra = self.n % self.num_threads;

        let seeds: Vec<u64> = (0..self.num_threads)
            .map(|_| ctx.rng().random())
            .collect();
        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();
        let op = &self.operator;

        let mut population = Population::new();

        std::thread::scope(|s| {
            let handles: Vec<_> = seeds
                .into_iter()
                .enumerate()
                .map(|(i, seed)| {
                    let reps = if i < extra { chunk_size + 1 } else { chunk_size };
                    s.spawn(move || -> Population<G, F> {
                        let mut rng = R::seed_from_u64(seed);
                        let mut thread_ctx = Context::new(fe, &mut rng, comp);
                        let mut pop = Population::new();
                        for _ in 0..reps {
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
