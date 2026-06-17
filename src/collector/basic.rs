//! A minimal collector that returns the final population and generation count.

use crate::collector::Collector;
use crate::core::population::Population;
use crate::core::state::State;

/// A minimal collector that produces a [`RunResult`] with no additional processing.
///
/// # Examples
///
/// ```
/// use evolve::{
///     algorithm::EvolutionaryAlgorithm,
///     collector::basic::Basic,
///     fitness::Maximize,
///     initialization::Random,
///     operators::sequential::combinator::Fill,
///     operators::sequential::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ea = EvolutionaryAlgorithm::new(
///     Random::new(),
///     MaxGenerations::new(10),
///     |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
///     Fill::from_population_size(RandomReset::new()),
///     NonZero::new(50).unwrap(),
///     rand::rng(),
///     Maximize,
/// );
///
/// let result = ea.run_with(Basic);
/// assert_eq!(result.generations(), 10);
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Basic;

impl Basic {
    /// Create a new `Basic` collector.
    pub fn new() -> Self {
        Self
    }
}

/// The result of a completed algorithm run.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunResult<G, F> {
    population: Population<G, F>,
    generations: usize,
}

impl<G, F> RunResult<G, F> {
    /// Get the final population.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Get the number of generations that were run.
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// Consume the result and return the population.
    pub fn into_population(self) -> Population<G, F> {
        self.population
    }
}

impl<G, F, Fe, C> Collector<G, F, Fe, C> for Basic {
    type Result = RunResult<G, F>;

    fn finalize(self, state: State<G, F>) -> Self::Result {
        RunResult {
            generations: state.generation(),
            population: state.into_population(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        algorithm::EvolutionaryAlgorithm,
        fitness::Maximize,
        initialization::Random,
        operators::sequential::{combinator::Fill, mutation::RandomReset},
        termination::MaxGenerations,
    };
    use std::num::NonZero;

    #[test]
    fn correct_generation_count() {
        let mut ga = EvolutionaryAlgorithm::new(
            Random::new(),
            MaxGenerations::new(10),
            |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
            Fill::from_population_size(RandomReset::new()),
            NonZero::new(20).unwrap(),
            rand::rng(),
            Maximize,
        );
        let result = ga.run_with(super::Basic::new());
        assert_eq!(result.generations(), 10);
    }

    #[test]
    fn population_is_preserved() {
        let mut ga = EvolutionaryAlgorithm::new(
            Random::new(),
            MaxGenerations::new(10),
            |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
            Fill::from_population_size(RandomReset::new()),
            NonZero::new(20).unwrap(),
            rand::rng(),
            Maximize,
        );
        let result = ga.run_with(super::Basic::new());
        assert_eq!(result.population().len(), 20);
    }
}
