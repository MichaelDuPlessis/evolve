use std::num::NonZero;

use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    fitness::FitnessComparator,
    operators::GeneticOperator,
};

/// Preserves the best `n` individuals from the current population.
///
/// When `n` is 1, returns the single best individual. For larger values,
/// uses partial sorting to efficiently find the top `n` without fully
/// sorting the population.
///
/// # Examples
///
/// ```
/// use evolve::operators::selection::Elitism;
/// use std::num::NonZero;
///
/// // Keep the single best individual each generation
/// let elite = Elitism::default();
///
/// // Keep the top 5
/// let elite = Elitism::new(NonZero::new(5).unwrap());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Elitism {
    amount: usize,
}

impl Elitism {
    /// Creates a new `Elitism` operator that preserves the best `n` individuals.
    pub fn new(amount: NonZero<usize>) -> Self {
        Self {
            amount: amount.get(),
        }
    }
}

impl Default for Elitism {
    fn default() -> Self {
        Self::new(NonZero::new(1).unwrap())
    }
}

impl<G, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Elitism
where
    F: PartialOrd + Clone,
    C: FitnessComparator<F>,
    G: Clone,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        if self.amount == 1 {
            Offspring::Single(state.population().best(ctx.comparator()).clone())
        } else {
            let mut refs: Vec<_> = state.population().iter().collect();
            refs.select_nth_unstable_by(self.amount - 1, |a, b| {
                if ctx.comparator().is_better(a.fitness(), b.fitness()) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            Offspring::Multiple(Population::from_iter(
                refs[..self.amount].iter().map(|i| (*i).clone()),
            ))
        }
    }
}
