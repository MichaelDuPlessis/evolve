use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};
use rand::{Rng, RngExt, SeedableRng};

/// Parallel version of [`Combine`](crate::operators::sequential::combinator::Combine).
///
/// Distributes a slice of operators across pool workers, each running on the same
/// input with its own RNG. Results are merged into one population.
///
/// Only available for homogeneous operator slices `[O]`. For heterogeneous
/// operator tuples, use the sequential [`Combine`](crate::operators::sequential::combinator::Combine).
#[derive(Debug, Clone)]
pub struct Combine<O: ?Sized> {
    operators: O,
}

impl<O> Combine<O> {
    /// Creates a new parallel `Combine`.
    pub fn new(operators: O) -> Self {
        Self { operators }
    }
}

macro_rules! impl_parallel_combine {
    ($container:ty) => {
        impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Combine<$container>
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
                let ops: &[O] = &self.operators;
                let inputs: vecpool::PoolVec<(u64, usize)> = (0..ops.len())
                    .map(|i| (ctx.rng().random::<u64>(), i))
                    .collect();

                let fe = ctx.fitness_evaluator();
                let comp = ctx.comparator();
                let runtime = ctx.runtime();

                let results = ctx.pool().map(&inputs, |(seed, idx)| {
                    let mut rng = R::seed_from_u64(*seed);
                    let mut thread_ctx = Context::new(fe, &mut rng, comp, runtime);
                    ops[*idx].apply(state, &mut thread_ctx).into_population()
                });

                let mut population = Population::new();
                for r in results {
                    population.merge(r.expect("pool task panicked"));
                }
                Offspring::Multiple(population)
            }
        }
    };
}

impl_parallel_combine!(Vec<O>);
impl_parallel_combine!(Box<[O]>);
impl_parallel_combine!([O]);
