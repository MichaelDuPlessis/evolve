use crate::core::{individual::Individual, population::Population};

#[derive(Debug)]
pub enum Offspring<G, F> {
    Single(Individual<G, F>),
    Multiple(Population<G, F>),
}

impl<G, F> Offspring<G, F> {
    /// Convert an `Offspring` into a `Population`.
    pub fn into_population(self) -> Population<G, F> {
        match self {
            Offspring::Single(individual) => Population::from_individuals(vec![individual]),
            Offspring::Multiple(population) => population,
        }
    }

    /// Get the number of `Individuals` in the `Offspring`.
    pub fn num_offspring(&self) -> usize {
        match self {
            Offspring::Single(_) => 1,
            Offspring::Multiple(population) => population.len(),
        }
    }
}
