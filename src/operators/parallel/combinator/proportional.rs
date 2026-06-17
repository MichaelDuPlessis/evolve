use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::{
        GeneticOperator,
        sequential::combinator::Fill as SequentialFill,
        sequential::combinator::fill::{FixedSize, GetSize, PopSize},
    },
};
use rand::{Rng, RngExt, SeedableRng};
use std::num::NonZero;

/// Parallel version of [`Proportional`](crate::operators::sequential::combinator::Proportional).
///
/// Each operator's portion runs as a separate pool task with its own RNG.
/// Results are merged into one population.
#[derive(Debug, Clone)]
pub struct Proportional<O, S = PopSize> {
    operators: O,
    size: S,
}

impl<O> Proportional<O> {
    /// Creates a new parallel `Proportional` that outputs the same size as the input population.
    pub fn new(operators: O) -> Self {
        Self {
            operators,
            size: PopSize::new(),
        }
    }
}

impl<O> Proportional<O, FixedSize> {
    /// Creates a new parallel `Proportional` with a fixed output size.
    pub fn with_size(operators: O, size: usize) -> Self {
        Self {
            operators,
            size: FixedSize::new(size),
        }
    }
}

impl<G, F, Fe, R, C, O, S> GeneticOperator<G, F, Fe, R, C> for Proportional<&[(O, NonZero<u16>)], S>
where
    O: GeneticOperator<G, F, Fe, R, C> + Sync,
    G: Send,
    F: Send,
    Fe: Sync,
    R: Rng + SeedableRng,
    C: Sync,
    S: GetSize<G, F>,
    State<G, F>: Sync,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let ops: &[(O, NonZero<u16>)] = self.operators;
        let target_size = self.size.get_size(state);
        let total_weight: u16 = ops.iter().map(|(_, w)| w.get()).sum();

        let mut targets: vecpool::PoolVec<usize> = vecpool::with_capacity(ops.len());
        let mut remaining = target_size;
        for (i, (_, weight)) in ops.iter().enumerate() {
            let target = if i == ops.len() - 1 {
                remaining
            } else {
                let t = (weight.get() as usize * target_size) / total_weight as usize;
                remaining -= t;
                t
            };
            targets.push(target);
        }

        let inputs: vecpool::PoolVec<(u64, usize)> = (0..ops.len())
            .map(|i| (ctx.rng().random::<u64>(), i))
            .collect();

        let fe = ctx.fitness_evaluator();
        let comp = ctx.comparator();
        let runtime = ctx.runtime();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let mut thread_ctx = Context::new(fe, &mut rng, comp, runtime);
            let fill = SequentialFill::from_fixed_size(&ops[*idx].0, targets[*idx]);
            fill.apply(state, &mut thread_ctx).into_population()
        });

        let mut population = Population::with_capacity(target_size);
        for r in results {
            population.merge(r.expect("pool task panicked"));
        }
        Offspring::Multiple(population)
    }
}
