use crate::fitness::FitnessEvaluation;

/// Individual with cached fitness
#[derive(Debug, Clone)]
pub struct Individual<G, F> {
    pub genome: G,
    pub fitness: Option<F>,
}

impl<G, F> Individual<G, F> {
    /// Create a new `Indivdual` from a genome.
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: None,
        }
    }

    /// Given a `FitnessEvaluation` calculate the `Individuals` fitness.
    pub fn calculate_fitness(&mut self, fitness_evaluation: &impl FitnessEvaluation<G, F>) {
        self.fitness = Some(fitness_evaluation.evaluate(&self.genome))
    }

    /// Sets the fitness value of an individual to None.
    pub fn invalidate_fitness(&mut self) {
        self.fitness = None;
    }
}
