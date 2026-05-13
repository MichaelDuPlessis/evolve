//! A minimal collector that returns the final population and generation count.

use crate::collector::Collector;
use crate::core::population::Population;
use crate::core::state::State;

/// A minimal collector that produces a [`RunResult`] with no additional processing.
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
