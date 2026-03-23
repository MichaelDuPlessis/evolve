use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};

/// Applies an operator `n` times and collects all results into one population.
///
/// # Examples
///
/// ```
/// use evolve::operators::combinator::Repeat;
/// use evolve::operators::mutation::RandomReset;
///
/// // Run mutation 5 times and merge the results
/// let op = Repeat::new(RandomReset::<u8>::new(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct Repeat<O> {
    operator: O,
    n: usize,
}

impl<O> Repeat<O> {
    /// Creates a new `Repeat` that applies the operator `n` times.
    pub fn new(operator: O, n: usize) -> Self {
        Self { operator, n }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Repeat<O>
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::new();
        for _ in 0..self.n {
            population.add_offspring(self.operator.apply(state, ctx));
        }

        Offspring::Multiple(population)
    }
}
