use crate::core::individual::Individual;

/// The population in the algorithm. It is a list of Individuals.
pub struct Population<G, F> {
    individuals: Vec<Individual<G, F>>,
}

impl<G, F> Population<G, F> {
    /// Create a new population from a `Vec` of `Individuals`.
    pub fn from_individuals(individuals: Vec<Individual<G, F>>) -> Self {
        Self { individuals }
    }

    /// Create a new population from a `Vec` of genomes.
    pub fn from_genomes(genomes: Vec<G>) -> Self {
        let individuals: Vec<Individual<G, F>> = genomes.into_iter().map(Individual::new).collect();
        Self::from_individuals(individuals)
    }

    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    pub fn individuals_mut(&mut self) -> &mut [Individual<G, F>] {
        &mut self.individuals
    }

    pub fn into_vec(self) -> Vec<Individual<G, F>> {
        self.individuals
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
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
