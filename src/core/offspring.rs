use crate::core::{individual::Individual, population::Population};

/// The result of applying a [`GeneticOperator`](crate::operators::GeneticOperator).
///
/// Operators may produce a single individual or an entire population. `Offspring`
/// captures both cases and can be converted into a [`Population`] when needed.
///
/// # Examples
///
/// ```
/// use evolve::core::{individual::Individual, offspring::Offspring, population::Population};
///
/// // A single offspring
/// let single = Offspring::<i32, i32>::Single(Individual::new(42));
/// assert_eq!(single.num_offspring(), 1);
///
/// // Convert to a population
/// let pop = single.into_population();
/// assert_eq!(pop.len(), 1);
/// ```
#[derive(Debug)]
pub enum Offspring<G, F> {
    /// A single individual produced by the operator.
    Single(Individual<G, F>),
    /// Multiple individuals produced by the operator.
    Multiple(Population<G, F>),
}

impl<G, F> Offspring<G, F> {
    /// Converts this `Offspring` into a [`Population`].
    pub fn into_population(self) -> Population<G, F> {
        match self {
            Offspring::Single(individual) => Population::from_individuals(vec![individual]),
            Offspring::Multiple(population) => population,
        }
    }

    /// Returns the number of individuals in this `Offspring`.
    pub fn num_offspring(&self) -> usize {
        match self {
            Offspring::Single(_) => 1,
            Offspring::Multiple(population) => population.len(),
        }
    }
}
