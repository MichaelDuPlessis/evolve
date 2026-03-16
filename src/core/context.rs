use crate::{core::population::Population, fitness::FitnessEvaluator, operators::GeneticOperator};

/// All the context needed for the genetic operators.
#[derive(Debug)]
pub struct Context<'a, Fe, R, C> {
    fitness: &'a Fe,
    rng: &'a mut R,
    comparator: &'a C,
}

impl<'a, Fe, R, C> Context<'a, Fe, R, C> {
    /// Create a new `Context`.
    pub fn new(fitness: &'a Fe, rng: &'a mut R, goal: &'a C) -> Self {
        Self {
            fitness,
            rng,
            comparator: goal,
        }
    }

    /// Get the `FitnessEvaluator`.
    pub fn fitness_evaluator(&self) -> &Fe {
        self.fitness
    }

    /// Get the `Rng` object.
    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    /// Get the goal for the problem.
    pub fn comparator(&self) -> &C {
        &self.comparator
    }
}

/// Contains the current state of the algorithm. This includes the current generation and the current population.
pub struct State<G, F> {
    population: Population<G, F>,
    generation: usize,
}

impl<G, F> State<G, F> {
    /// Create a new `State`.
    pub fn new(population: Population<G, F>, generation: usize) -> Self {
        Self {
            population,
            generation,
        }
    }

    /// Create a new `State` while keeping the generation the same.
    pub fn with_population(&self, population: Population<G, F>) -> Self {
        Self::new(population, self.generation)
    }

    /// Get the `Population`.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Get the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Increase the generation that the `State` is on.
    pub(crate) fn inc_generation(&mut self) {
        self.generation += 1;
    }

    /// Applies the operators to the `State` and updates the population.
    pub(crate) fn apply_operators<Fe, R, C>(
        &mut self,
        ctx: &mut Context<Fe, R, C>,
        ops: &mut impl GeneticOperator<G, F, Fe, R, C>,
    ) where
        Fe: FitnessEvaluator<G, F>,
    {
        self.population = ops.apply(self, ctx);
    }

    /// Transforms the `State` into a `Population`.
    pub(crate) fn into_population(self) -> Population<G, F> {
        self.population
    }
}
