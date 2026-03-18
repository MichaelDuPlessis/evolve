use rand::{Rng, seq::IndexedRandom};

use crate::{
    core::{individual::Individual, offspring::Offpring},
    fitness::FitnessComparator,
};
use std::slice::ChunksExact;

/// The population in the algorithm. It is a list of Individuals.
#[derive(Debug)]
pub struct Population<G, F> {
    individuals: Vec<Individual<G, F>>,
}

impl<G, F> Population<G, F> {
    /// Create a new empty population.
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// Create an empty population with a certain capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            individuals: Vec::with_capacity(capacity),
        }
    }

    /// Create a new population from a `Vec` of `Individuals`.
    pub fn from_individuals(individuals: Vec<Individual<G, F>>) -> Self {
        Self { individuals }
    }

    /// Extend a `Population` with another `Populatin` or a list of `Individuals`.
    pub fn extend(&mut self, individuals: impl IntoIterator<Item = Individual<G, F>>) {
        self.individuals.extend(individuals);
    }

    /// Number of individuals in the population.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Returns true if the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Immutable iterator over individuals.
    pub fn iter(&self) -> std::slice::Iter<'_, Individual<G, F>> {
        self.individuals.iter()
    }

    /// Mutable iterator over individuals.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Individual<G, F>> {
        self.individuals.iter_mut()
    }

    /// Create chunks of exactly the size specified.
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, Individual<G, F>> {
        self.individuals.chunks_exact(chunk_size)
    }

    /// Returns the population as a slice.
    pub fn as_slice(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    /// Returns the population as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [Individual<G, F>] {
        &mut self.individuals
    }

    /// Consumes the population and returns the inner Vec.
    pub fn into_vec(self) -> Vec<Individual<G, F>> {
        self.individuals
    }

    /// Choose a random individual from the population.
    pub fn choose<R: Rng>(&self, rng: &mut R) -> Option<&Individual<G, F>> {
        self.individuals.choose(rng)
    }

    /// Get the best individual in the Population
    pub fn best<C>(&self, comparator: &C) -> &Individual<G, F>
    where
        F: PartialOrd,
        C: FitnessComparator<F>,
    {
        self.individuals
            .iter()
            .reduce(|a, b| {
                if comparator.is_better(a.fitness(), b.fitness()) {
                    a
                } else {
                    b
                }
            })
            .expect("population cannot be empty")
    }

    /// Add a new `Individual` to the population
    pub fn add(&mut self, individual: Individual<G, F>) {
        self.individuals.push(individual);
    }

    /// Merges another `Population` into one this one.
    pub fn merge(&mut self, population: Population<G, F>) {
        self.individuals.extend(population.individuals);
    }

    /// Add an `Offspring` to the population.
    pub fn add_offspring(&mut self, offspring: Offpring<G, F>) {
        match offspring {
            Offpring::Single(individual) => self.add(individual),
            Offpring::Multiple(population) => self.merge(population),
        }
    }
}

impl<G, F> From<Vec<Individual<G, F>>> for Population<G, F> {
    fn from(value: Vec<Individual<G, F>>) -> Self {
        Self::from_individuals(value)
    }
}

impl<G, F> From<Offpring<G, F>> for Population<G, F> {
    fn from(value: Offpring<G, F>) -> Self {
        value.into_population()
    }
}

impl<G, F> IntoIterator for Population<G, F> {
    type Item = Individual<G, F>;
    type IntoIter = std::vec::IntoIter<Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.into_iter()
    }
}

impl<'a, G, F> IntoIterator for &'a Population<G, F> {
    type Item = &'a Individual<G, F>;
    type IntoIter = std::slice::Iter<'a, Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.iter()
    }
}

impl<'a, G, F> IntoIterator for &'a mut Population<G, F> {
    type Item = &'a mut Individual<G, F>;
    type IntoIter = std::slice::IterMut<'a, Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.iter_mut()
    }
}

impl<G, F> FromIterator<Individual<G, F>> for Population<G, F> {
    fn from_iter<T: IntoIterator<Item = Individual<G, F>>>(iter: T) -> Self {
        Self {
            individuals: iter.into_iter().collect(),
        }
    }
}
