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
