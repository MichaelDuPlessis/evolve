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
/// # #[cfg(not(feature = "parallel"))]
/// let ctx = Context::new(&fitness_fn, &mut rng, &Maximize);
/// # #[cfg(feature = "parallel")]
/// # let runtime = pooled::Runtime::new(1);
/// # #[cfg(feature = "parallel")]
/// # let ctx = Context::new(&fitness_fn, &mut rng, &Maximize, &runtime);
/// ```
pub struct Context<'a, Fe, R, C> {
    fitness: &'a Fe,
    rng: &'a mut R,
    comparator: &'a C,
    #[cfg(feature = "parallel")]
    runtime: &'a pooled::Runtime,
}

impl<'a, Fe, R, C> Context<'a, Fe, R, C> {
    /// Create a new `Context`.
    #[cfg(not(feature = "parallel"))]
    pub fn new(fitness: &'a Fe, rng: &'a mut R, comparator: &'a C) -> Self {
        Self {
            fitness,
            rng,
            comparator,
        }
    }

    /// Create a new `Context` with a thread pool runtime for parallel operations.
    #[cfg(feature = "parallel")]
    pub fn new(
        fitness: &'a Fe,
        rng: &'a mut R,
        comparator: &'a C,
        runtime: &'a pooled::Runtime,
    ) -> Self {
        Self {
            fitness,
            rng,
            comparator,
            runtime,
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

    /// Get the thread pool for parallel operations.
    #[cfg(feature = "parallel")]
    pub fn pool(&self) -> pooled::map::MapPool<'_> {
        self.runtime.map_pool()
    }

    /// Get the runtime for parallel operations.
    #[cfg(feature = "parallel")]
    pub fn runtime(&self) -> &pooled::Runtime {
        self.runtime
    }
}

impl<'a, Fe, R, C> std::fmt::Debug for Context<'a, Fe, R, C>
where
    Fe: std::fmt::Debug,
    R: std::fmt::Debug,
    C: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("fitness", &self.fitness)
            .field("rng", &self.rng)
            .field("comparator", &self.comparator)
            .finish_non_exhaustive()
    }
}
