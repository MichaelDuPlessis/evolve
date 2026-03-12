use std::slice::ChunksExact;

use crate::core::individual::Individual;

/// The population in the algorithm. It is a list of Individuals.
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

    /// Create a new population from a `Vec` of genomes.
    pub fn from_genomes(genomes: Vec<G>) -> Self {
        let individuals: Vec<Individual<G, F>> = genomes.into_iter().map(Individual::new).collect();
        Self::from_individuals(individuals)
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

    /// Get the best individual in the Population
    pub fn best(self) -> Individual<G, F>
    where
        F: PartialOrd,
    {
        self.individuals
            .into_iter()
            .max_by(|a, b| {
                a.fitness
                    .as_ref()
                    .unwrap()
                    .partial_cmp(b.fitness.as_ref().unwrap())
                    .unwrap()
            })
            .expect("population cannot be empty")
    }

    /// Add a new individual into the population
    pub fn add(&mut self, individual: Individual<G, F>) {
        self.individuals.push(individual);
    }
}

impl<G, F> From<Vec<Individual<G, F>>> for Population<G, F> {
    fn from(value: Vec<Individual<G, F>>) -> Self {
        Self::from_individuals(value)
    }
}

impl<G, F> From<Vec<G>> for Population<G, F> {
    fn from(value: Vec<G>) -> Self {
        Self::from_genomes(value)
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
