use crate::core::{individual::Individual, population::Population};

pub enum Offpring<G, F> {
    Single(Individual<G, F>),
    Multiple(Population<G, F>),
}

impl<G, F> Offpring<G, F> {
    /// Convert an `Offspring` into a `Population`.
    pub fn into_population(self) -> Population<G, F> {
        match self {
            Offpring::Single(individual) => Population::from_individuals(vec![individual]),
            Offpring::Multiple(population) => population,
        }
    }
}
