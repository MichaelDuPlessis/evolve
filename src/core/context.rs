/// All the context needed by genetic operators during execution.
///
/// Bundles references to the fitness evaluator, random number generator, and
/// fitness comparator so operators can evaluate new individuals, generate
/// random values, and compare fitness without owning these resources.
///
/// # Examples
///
/// ```
/// use evolve::core::context::Context;
/// use evolve::fitness::Maximize;
///
/// let fitness_fn = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
/// let mut rng = rand::rng();
/// let ctx = Context::new(&fitness_fn, &mut rng, &Maximize);
/// ```
#[derive(Debug)]
pub struct Context<'a, Fe, R, C> {
    fitness: &'a Fe,
    rng: &'a mut R,
    comparator: &'a C,
}

impl<'a, Fe, R, C> Context<'a, Fe, R, C> {
    /// Create a new `Context`.
    pub fn new(fitness: &'a Fe, rng: &'a mut R, comparator: &'a C) -> Self {
        Self {
            fitness,
            rng,
            comparator,
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
        self.comparator
    }
}
