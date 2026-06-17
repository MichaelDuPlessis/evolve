use crate::core::{context::Context, offspring::Offspring, population::Population, state::State};
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt};

/// Applies an inner operator to each individual with a given probability.
///
/// Individuals not selected by the rate check pass through unchanged,
/// preserving their cached fitness.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::with_rate::WithRate;
/// use evolve::operators::sequential::mutation::RandomReset;
///
/// // Apply mutation to each individual with 10% probability
/// let op = WithRate::new(RandomReset::<u8>::new(), 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct WithRate<O> {
    operator: O,
    rate: f64,
}

impl<O> WithRate<O> {
    /// Creates a new `WithRate` wrapper.
    ///
    /// # Panics
    /// Panics if `rate` is not in `0.0..=1.0`.
    pub fn new(operator: O, rate: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&rate),
            "rate must be between 0.0 and 1.0"
        );
        Self { operator, rate }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for WithRate<O>
where
    G: Clone,
    F: Clone,
    R: Rng,
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::with_capacity(state.population().len());

        for ind in state.population() {
            if ctx.rng().random::<f64>() < self.rate {
                let single_pop: Population<G, F> =
                    std::iter::once(ind.clone_genome_only()).collect();
                let single_state = state.with_population(single_pop);
                let result = self.operator.apply(&single_state, ctx);
                match result {
                    Offspring::Single(i) => population.add(i),
                    Offspring::Multiple(p) => {
                        if let Some(i) = p.into_iter().next() {
                            population.add(i);
                        }
                    }
                }
            } else {
                population.add(ind.clone());
            }
        }

        Offspring::Multiple(population)
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let generation = state.generation();
        let mut population = Population::with_capacity(state.population().len());

        for ind in state.into_population() {
            if ctx.rng().random::<f64>() < self.rate {
                let single_pop: Population<G, F> = std::iter::once(ind).collect();
                let single_state = State::new(single_pop, generation);
                let result = self.operator.transform(single_state, ctx);
                match result {
                    Offspring::Single(i) => population.add(i),
                    Offspring::Multiple(p) => {
                        if let Some(i) = p.into_iter().next() {
                            population.add(i);
                        }
                    }
                }
            } else {
                population.add(ind);
            }
        }

        Offspring::Multiple(population)
    }
}
