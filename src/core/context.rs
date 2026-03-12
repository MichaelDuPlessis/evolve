use crate::core::population::Population;

/// All the context needed for the genetic operators.
pub struct Context<'a, G, F, Fe> {
    population: &'a Population<G, F>,
    fitness: &'a Fe,
    generation: usize,
}

impl<'a, G, F, Fe> Context<'a, G, F, Fe> {
    /// Create a new `Context`.
    pub fn new(population: &'a Population<G, F>, fitness: &'a Fe, generation: usize) -> Self {
        Self {
            population,
            fitness,
            generation,
        }
    }

    /// Get the `Population`.
    pub fn population(&self) -> &Population<G, F> {
        self.population
    }

    /// Get the `FitnessEvaluation`.
    pub fn fitness(&self) -> &Fe {
        self.fitness
    }

    /// Get the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }
}
