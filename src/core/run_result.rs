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
///     algorithm::ga::GeneticAlgorithm,
///     fitness::Maximize,
///     initialization::Random,
///     operators::combinator::Fill,
///     operators::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ga = GeneticAlgorithm::new(
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
/// let best = result.population.best(&Maximize);
/// println!("generations: {}, best fitness: {:?}", result.generations, best.fitness());
/// ```
#[derive(Debug)]
pub struct RunResult<G, F> {
    pub population: Population<G, F>,
    pub generations: usize,
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
