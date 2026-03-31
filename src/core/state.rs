use crate::{
    core::{context::Context, population::Population},
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};

/// The current state of the algorithm, holding the population and generation counter.
///
/// Passed to [`GeneticOperator::apply`] so
/// operators can read the current population and generation number.
#[derive(Debug)]
pub struct State<G, F> {
    population: Population<G, F>,
    generation: usize,
}

impl<G, F> State<G, F> {
    /// Create a new `State`.
    pub fn new(population: Population<G, F>, generation: usize) -> Self {
        Self {
            population,
            generation,
        }
    }

    /// Create a new `State` while keeping the generation the same.
    pub fn with_population(&self, population: Population<G, F>) -> Self {
        Self::new(population, self.generation)
    }

    /// Get the `Population`.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Get the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Increase the generation that the `State` is on.
    pub(crate) fn inc_generation(&mut self) {
        self.generation += 1;
    }

    /// Applies the operators to the `State` and updates the population.
    pub(crate) fn apply_operators<Fe, R, C>(
        &mut self,
        ctx: &mut Context<Fe, R, C>,
        ops: &mut impl GeneticOperator<G, F, Fe, R, C>,
    ) where
        Fe: FitnessEvaluator<G, F>,
    {
        self.population = ops.apply(self, ctx).into();
    }

    /// Transforms the `State` into a `Population`.
    pub(crate) fn into_population(self) -> Population<G, F> {
        self.population
    }
}
