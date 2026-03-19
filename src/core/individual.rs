use crate::fitness::FitnessEvaluator;

/// A single candidate solution paired with its fitness value.
///
/// An `Individual` wraps a genome of type `G` and eagerly evaluates its fitness
/// using a [`FitnessEvaluator`] at construction time.
///
/// # Examples
///
/// ```
/// use evolve::core::individual::Individual;
///
/// let ind = Individual::new([1u32, 2, 3], &|g: &[u32; 3]| g.iter().sum::<u32>());
/// assert_eq!(*ind.genome(), [1, 2, 3]);
/// assert_eq!(*ind.fitness(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct Individual<G, F> {
    genome: G,
    fitness: F,
}

impl<G, F> Individual<G, F> {
    /// Creates a new `Individual` by evaluating the fitness of the given genome.
    pub fn new<Fe>(genome: G, fitness_evaluator: &Fe) -> Self
    where
        Fe: FitnessEvaluator<G, F>,
    {
        let fitness = fitness_evaluator.evaluate(&genome);
        Self { genome, fitness }
    }

    /// Returns a reference to the genome.
    pub fn genome(&self) -> &G {
        &self.genome
    }

    /// Returns a reference to the fitness value.
    pub fn fitness(&self) -> &F {
        &self.fitness
    }
}
