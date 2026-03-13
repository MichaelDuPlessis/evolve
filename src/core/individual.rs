use crate::fitness::FitnessEvaluator;

/// Individual with cached fitness
#[derive(Debug, Clone)]
pub struct Individual<G, F> {
    genome: G,
    fitness: F,
}

impl<G, F> Individual<G, F> {
    /// Create a new `Indivdual` from a genome and a `FitnessEvaluator`.
    pub fn new<Fe>(genome: G, fitness_evaluator: &Fe) -> Self
    where
        Fe: FitnessEvaluator<G, F>,
    {
        let fitness = fitness_evaluator.evaluate(&genome);
        Self { genome, fitness }
    }

    /// Get the genome.
    pub fn genome(&self) -> &G {
        &self.genome
    }

    /// Get the fitness.
    pub fn fitness(&self) -> &F {
        &self.fitness
    }
}
