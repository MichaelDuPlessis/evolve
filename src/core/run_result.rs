use crate::core::{population::Population, state::State};

/// The result of running a genetic algorithm.
///
/// Contains the final population and the number of generations that were executed.
/// Use [`Population::best`] to retrieve the best individual.
///
/// # Examples
///
/// ```
/// use evolve::{
///     algorithm::EvolutionaryAlgorithm,
///     fitness::Maximize,
///     initialization::Random,
///     operators::sequential::combinator::Fill,
///     operators::sequential::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ga = EvolutionaryAlgorithm::new(
///     Random::new(),
///     MaxGenerations::new(10),
///     |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
///     Fill::from_population_size(RandomReset::new()),
///     NonZero::new(50).unwrap(),
///     rand::rng(),
///     Maximize,
/// );
///
/// let result = ga.run();
/// let fe = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
/// let best = result.population().best(&fe, &Maximize);
/// println!("generations: {}, best fitness: {:?}", result.generations(), best.fitness(&fe));
/// ```
#[derive(Debug)]
pub struct RunResult<G, F> {
    population: Population<G, F>,
    generations: usize,
}

impl<G, F> RunResult<G, F> {
    /// Returns a reference to the final population.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Returns the number of generations that were run.
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// Consumes the result and returns the final population.
    pub fn into_population(self) -> Population<G, F> {
        self.population
    }
}

impl<G, F> From<State<G, F>> for RunResult<G, F> {
    fn from(value: State<G, F>) -> Self {
        let generations = value.generation();
        let population = value.into_population();

        Self {
            population,
            generations,
        }
    }
}
