use crate::fitness::FitnessEvaluator;

#[cfg(feature = "parallel")]
use std::sync::OnceLock as FitnessCell;

#[cfg(not(feature = "parallel"))]
use std::cell::OnceCell as FitnessCell;

/// A single candidate solution with lazily evaluated fitness.
///
/// An `Individual` wraps a genome of type `G` and lazily evaluates its fitness
/// using a [`FitnessEvaluator`] on first access.
///
/// # Examples
///
/// ```
/// use evolve::core::individual::Individual;
///
/// let ind = Individual::<[u32; 3], u32>::new([1, 2, 3]);
/// assert_eq!(*ind.genome(), [1, 2, 3]);
/// assert_eq!(*ind.fitness(&|g: &[u32; 3]| g.iter().sum::<u32>()), 6);
/// ```
#[derive(Debug, Clone)]
pub struct Individual<G, F> {
    genome: G,
    fitness: FitnessCell<F>,
}

impl<G, F> Individual<G, F> {
    /// Creates a new `Individual` with the given genome and unevaluated fitness.
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: FitnessCell::new(),
        }
    }

    /// Creates a new `Individual` with a pre-computed fitness value.
    pub fn from_parts(genome: G, fitness: F) -> Self {
        Self {
            genome,
            fitness: FitnessCell::from(fitness),
        }
    }

    /// Returns the fitness value if it has already been evaluated.
    pub fn try_fitness(&self) -> Option<&F> {
        self.fitness.get()
    }

    /// Returns a reference to the genome.
    pub fn genome(&self) -> &G {
        &self.genome
    }

    /// Returns a reference to the fitness value, evaluating it on first access.
    pub fn fitness<Fe>(&self, fitness_evaluator: &Fe) -> &F
    where
        Fe: FitnessEvaluator<G, F>,
    {
        self.fitness
            .get_or_init(|| fitness_evaluator.evaluate(&self.genome))
    }

    /// Creates a new `Individual` with a clone of this individual's genome
    /// and an unevaluated fitness cell.
    ///
    /// Use this when the cloned individual will have its genome modified
    /// (e.g., by crossover or mutation), making the original fitness irrelevant.
    pub fn clone_genome_only(&self) -> Self
    where
        G: Clone,
    {
        Self {
            genome: self.genome.clone(),
            fitness: FitnessCell::new(),
        }
    }

    /// Consumes this individual, applies a function to its genome, and returns
    /// a new individual with the modified genome and unevaluated fitness.
    pub fn mutate_genome(self, f: impl FnOnce(&mut G)) -> Self {
        let mut genome = self.genome;
        f(&mut genome);
        Self::new(genome)
    }
}

#[cfg(feature = "serde")]
impl<G, F> serde::Serialize for Individual<G, F>
where
    G: serde::Serialize,
    F: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Individual", 2)?;
        s.serialize_field("genome", &self.genome)?;
        s.serialize_field("fitness", &self.try_fitness())?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, G, F> serde::Deserialize<'de> for Individual<G, F>
where
    G: serde::Deserialize<'de>,
    F: serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Data<G, F> {
            genome: G,
            fitness: Option<F>,
        }
        let data = Data::<G, F>::deserialize(deserializer)?;
        Ok(match data.fitness {
            Some(f) => Self::from_parts(data.genome, f),
            None => Self::new(data.genome),
        })
    }
}
