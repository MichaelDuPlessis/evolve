//! Population initialization strategies.
//!
//! Defines the [`Initializer`] trait and provides [`Random`], which generates
//! a population of random genomes.

use crate::{
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluator,
    random::Randomizable,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::Range;

/// Creates the initial population for the algorithm.
///
/// # Examples
///
/// ```
/// use evolve::core::context::Context;
/// use evolve::fitness::Maximize;
/// use evolve::initialization::{Initializer, Random};
/// use std::num::NonZero;
///
/// let mut rng = rand::rng();
/// let fitness_fn = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
/// # #[cfg(not(feature = "parallel"))]
/// let mut ctx = Context::new(&fitness_fn, &mut rng, &Maximize);
/// # #[cfg(feature = "parallel")]
/// # let runtime = pooled::Runtime::new(1);
/// # #[cfg(feature = "parallel")]
/// # let mut ctx = Context::new(&fitness_fn, &mut rng, &Maximize, &runtime);
///
/// let pop = Random::new().initialize(NonZero::new(100).unwrap(), &mut ctx);
/// assert_eq!(pop.len(), 100);
/// ```
pub trait Initializer<G, F, Fe, R, C>
where
    Fe: FitnessEvaluator<G, F>,
{
    /// Creates the initial population of the given size.
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F>;
}

/// An [`Initializer`] that creates a population of random genomes.
///
/// Requires the genome type to implement [`Randomizable`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Random;

impl Random {
    /// Creates a new `Random` initializer.
    pub fn new() -> Self {
        Self
    }
}

impl<G, F, Fe, R, C> Initializer<G, F, Fe, R, C> for Random
where
    Fe: FitnessEvaluator<G, F>,
    G: Randomizable<R>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F> {
        (0..population_size.get())
            .map(|_| Individual::new(G::random(ctx.rng())))
            .collect()
    }
}

/// An [`Initializer`] that creates a population of random variable-length genomes.
///
/// Each genome's length is sampled uniformly from the given range.
///
/// # Examples
///
/// ```
/// use evolve::core::context::Context;
/// use evolve::fitness::Maximize;
/// use evolve::initialization::{Initializer, RangedRandom};
/// use std::num::NonZero;
///
/// let mut rng = rand::rng();
/// let fitness_fn = |g: &Vec<u8>| g.iter().map(|&x| x as u32).sum::<u32>();
/// # #[cfg(not(feature = "parallel"))]
/// let mut ctx = Context::new(&fitness_fn, &mut rng, &Maximize);
/// # #[cfg(feature = "parallel")]
/// # let runtime = pooled::Runtime::new(1);
/// # #[cfg(feature = "parallel")]
/// # let mut ctx = Context::new(&fitness_fn, &mut rng, &Maximize, &runtime);
///
/// let pop = RangedRandom::<u8>::new(5..10).initialize(NonZero::new(50).unwrap(), &mut ctx);
/// assert_eq!(pop.len(), 50);
/// for ind in &pop {
///     assert!((5..10).contains(&ind.genome().len()));
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RangedRandom<T> {
    length_range: Range<usize>,
    _marker: PhantomData<T>,
}

impl<T> RangedRandom<T> {
    /// Creates a new `RangedRandom` initializer with the given length range.
    ///
    /// # Panics
    ///
    /// Panics if `length_range` is empty.
    pub fn new(length_range: Range<usize>) -> Self {
        assert!(!length_range.is_empty(), "length_range must be non-empty");
        Self {
            length_range,
            _marker: PhantomData,
        }
    }
}

impl<T, F, Fe, R, C> Initializer<Vec<T>, F, Fe, R, C> for RangedRandom<T>
where
    Fe: FitnessEvaluator<Vec<T>, F>,
    T: Randomizable<R>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<Vec<T>, F> {
        (0..population_size.get())
            .map(|_| {
                let len = ctx.rng().random_range(self.length_range.clone());
                let genome: Vec<T> = (0..len).map(|_| T::random(ctx.rng())).collect();
                Individual::new(genome)
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::context::Context;
    use crate::fitness::Maximize;

    fn id(g: &[u8; 2]) -> u16 {
        g[0] as u16 + g[1] as u16
    }

    #[cfg(feature = "parallel")]
    fn test_runtime() -> &'static pooled::Runtime {
        use std::sync::LazyLock;
        static RUNTIME: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        &RUNTIME
    }

    #[test]
    fn random_initializer_creates_correct_size() {
        let mut rng = rand::rng();
        #[cfg(not(feature = "parallel"))]
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        #[cfg(feature = "parallel")]
        let mut ctx = Context::new(
            &(id as fn(&[u8; 2]) -> u16),
            &mut rng,
            &Maximize,
            test_runtime(),
        );
        let pop = Random::new().initialize(NonZero::new(10).unwrap(), &mut ctx);
        assert_eq!(pop.len(), 10);
    }

    #[test]
    fn random_initializer_evaluates_fitness() {
        let mut rng = rand::rng();
        #[cfg(not(feature = "parallel"))]
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        #[cfg(feature = "parallel")]
        let mut ctx = Context::new(
            &(id as fn(&[u8; 2]) -> u16),
            &mut rng,
            &Maximize,
            test_runtime(),
        );
        let pop = Random::new().initialize(NonZero::new(5).unwrap(), &mut ctx);
        for ind in &pop {
            assert_eq!(
                *ind.fitness(&id),
                ind.genome()[0] as u16 + ind.genome()[1] as u16
            );
        }
    }

    fn vec_sum(g: &Vec<u8>) -> u32 {
        g.iter().map(|&x| x as u32).sum()
    }

    #[test]
    fn ranged_random_creates_correct_size() {
        let mut rng = rand::rng();
        #[cfg(not(feature = "parallel"))]
        let mut ctx = Context::new(&(vec_sum as fn(&Vec<u8>) -> u32), &mut rng, &Maximize);
        #[cfg(feature = "parallel")]
        let mut ctx = Context::new(
            &(vec_sum as fn(&Vec<u8>) -> u32),
            &mut rng,
            &Maximize,
            test_runtime(),
        );
        let pop = RangedRandom::<u8>::new(3..8).initialize(NonZero::new(20).unwrap(), &mut ctx);
        assert_eq!(pop.len(), 20);
    }

    #[test]
    fn ranged_random_genome_lengths_within_range() {
        let mut rng = rand::rng();
        #[cfg(not(feature = "parallel"))]
        let mut ctx = Context::new(&(vec_sum as fn(&Vec<u8>) -> u32), &mut rng, &Maximize);
        #[cfg(feature = "parallel")]
        let mut ctx = Context::new(
            &(vec_sum as fn(&Vec<u8>) -> u32),
            &mut rng,
            &Maximize,
            test_runtime(),
        );
        let pop = RangedRandom::<u8>::new(5..10).initialize(NonZero::new(50).unwrap(), &mut ctx);
        for ind in &pop {
            assert!((5..10).contains(&ind.genome().len()));
        }
    }
}
