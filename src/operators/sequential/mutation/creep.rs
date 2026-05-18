use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::fitness::FitnessEvaluator;
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt};

/// Adds a small random offset to a random gene.
///
/// The discrete equivalent of Gaussian mutation. Each application picks one
/// gene and adds a uniformly sampled offset in `[-step, +step]`.
/// Uses wrapping arithmetic to handle overflow, unless bounds are set
/// in which case the result is clamped.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::Creep;
///
/// // Unbounded (wrapping)
/// let mutation = Creep::<i32>::new(5);
///
/// // Lower bound only
/// let mutation = Creep::<i32>::new(5).min(0);
///
/// // Both bounds
/// let mutation = Creep::<i32>::new(5).min(0).max(100);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Creep<T> {
    step: T,
    min: Option<T>,
    max: Option<T>,
}

impl<T> Creep<T> {
    /// Creates a new `Creep` mutation with the given maximum step size.
    pub fn new(step: T) -> Self {
        Self { step, min: None, max: None }
    }

    /// Sets the lower bound. Genes are clamped to this minimum after mutation.
    pub fn min(mut self, min: T) -> Self {
        self.min = Some(min);
        self
    }

    /// Sets the upper bound. Genes are clamped to this maximum after mutation.
    pub fn max(mut self, max: T) -> Self {
        self.max = Some(max);
        self
    }
}

/// Trait for integer types that support creep mutation.
pub(crate) trait CreepMutate: Copy {
    fn creep(self, step: Self, rng: &mut impl Rng) -> Self;
}

macro_rules! impl_creep_signed {
    ($($t:ty),*) => {
        $(
            impl CreepMutate for $t {
                fn creep(self, step: Self, rng: &mut impl Rng) -> Self {
                    let offset = rng.random_range(-step..=step);
                    self.wrapping_add(offset)
                }
            }
        )*
    };
}

macro_rules! impl_creep_unsigned {
    ($($t:ty, $signed:ty);*) => {
        $(
            impl CreepMutate for $t {
                fn creep(self, step: Self, rng: &mut impl Rng) -> Self {
                    let offset = rng.random_range(-(step as $signed)..=(step as $signed));
                    self.wrapping_add(offset as $t)
                }
            }
        )*
    };
}

impl_creep_signed!(i8, i16, i32, i64);
impl_creep_unsigned!(u8, i16; u16, i32; u32, i64; u64, i128);

trait CreepCollection {}
impl<T> CreepCollection for Vec<T> {}
impl<T> CreepCollection for Box<[T]> {}
impl<T> CreepCollection for [T] {}
impl<T, const N: usize> CreepCollection for [T; N] {}

impl<G, T, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Creep<T>
where
    G: Clone + AsMut<[T]> + CreepCollection,
    T: CreepMutate + Ord,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = individual.genome().clone();
            let genes = genome.as_mut();
            let idx = ctx.rng().random_range(0..genes.len());
            genes[idx] = genes[idx].creep(self.step, ctx.rng());
            if let Some(min) = self.min { genes[idx] = std::cmp::max(genes[idx], min); }
            if let Some(max) = self.max { genes[idx] = std::cmp::min(genes[idx], max); }
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let step = self.step;
        let min = self.min;
        let max = self.max;
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let genes = genome.as_mut();
                    let idx = ctx.rng().random_range(0..genes.len());
                    genes[idx] = genes[idx].creep(step, ctx.rng());
                    if let Some(lo) = min { genes[idx] = std::cmp::max(genes[idx], lo); }
                    if let Some(hi) = max { genes[idx] = std::cmp::min(genes[idx], hi); }
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}
