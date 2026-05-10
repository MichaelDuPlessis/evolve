use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};
use rand::{Rng, RngExt, SeedableRng};

/// Parallel version of [`Repeat`](crate::operators::sequential::combinator::Repeat).
///
/// Distributes `n` repetitions across pool workers, each with its own RNG.
/// Results are merged into one population.
#[derive(Debug, Clone)]
pub struct Repeat<O> {
    operator: O,
    n: usize,
}

impl<O> Repeat<O> {
    /// Creates a new parallel `Repeat`.
    pub fn new(operator: O, n: usize) -> Self {
        Self { operator, n }
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
        let num_chunks = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let chunk_size = self.n / num_chunks;
        let extra = self.n % num_chunks;

        let inputs: Vec<(u64, usize)> = (0..num_chunks)
            .map(|i| {
                let reps = chunk_size + if i < extra { 1 } else { 0 };
                (ctx.rng().random::<u64>(), reps)
            })
            .collect();

        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();
        let op = &self.operator;
        let runtime = ctx.runtime();

        let results = ctx.pool().map(&inputs, |(seed, reps)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut thread_ctx = Context::new(fe, &mut rng, comp, runtime);
            let mut pop = Population::new();
            for _ in 0..*reps {
                pop.add_offspring(op.apply(state, &mut thread_ctx));
            }
            pop
        });

        let mut population = Population::new();
        for r in results {
            population.merge(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
