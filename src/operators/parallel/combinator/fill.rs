use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::{GeneticOperator, sequential::combinator::Fill as SequentialFill},
};
use rand::{Rng, RngExt, SeedableRng};

/// Parallel version of [`Fill`](crate::operators::sequential::combinator::Fill).
///
/// Splits the target size across pool workers, each running a sequential
/// [`Fill`](crate::operators::sequential::combinator::Fill) with its own RNG.
/// Results are merged into a single population.
#[derive(Debug, Clone)]
pub struct Fill<O> {
    operator: O,
    target_size: usize,
}

impl<O> Fill<O> {
    /// Creates a new parallel `Fill`.
    pub fn new(operator: O, target_size: usize) -> Self {
        Self {
            operator,
            target_size,
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
        let num_chunks = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let chunk_size = self.target_size / num_chunks;
        let extra = self.target_size % num_chunks;

        let inputs: Vec<(u64, usize)> = (0..num_chunks)
            .map(|i| {
                let size = chunk_size + if i < extra { 1 } else { 0 };
                (ctx.rng().random::<u64>(), size)
            })
            .collect();

        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();
        let op = &self.operator;
        let runtime = ctx.runtime();

        let results = ctx.pool().map(&inputs, |(seed, size)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut thread_ctx = Context::new(fe, &mut rng, comp, runtime);
            let fill = SequentialFill::from_fixed_size(op, *size);
            fill.apply(state, &mut thread_ctx).into_population()
        });

        let mut population = Population::with_capacity(self.target_size);
        for r in results {
            population.merge(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
