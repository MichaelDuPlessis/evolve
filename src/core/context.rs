use crate::core::population::Population;

/// All the context needed for the genetic operators.
pub struct Context<'a, Fe, R> {
    fitness: &'a Fe,
    rng: &'a mut R,
}

impl<'a, Fe, R> Context<'a, Fe, R> {
    /// Create a new `Context`.
    pub fn new(fitness: &'a Fe, rng: &'a mut R) -> Self {
        Self { fitness, rng }
    }

    /// Get the `FitnessEvaluation`.
    pub fn fitness(&self) -> &Fe {
        self.fitness
    }
}

/// Contains the current state of the algorithm. This includes the current generation and the current population.
pub struct State<G, F> {
    pub population: Population<G, F>,
    pub generation: usize,
}

impl<G, F> State<G, F> {
    /// Create a new `State`.
    pub fn new(population: Population<G, F>, generation: usize) -> Self {
        Self {
            population,
            generation,
        }
    }
    /// Get the `Population`.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Get the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }
}
