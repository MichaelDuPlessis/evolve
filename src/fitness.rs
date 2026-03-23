//! Fitness evaluation and comparison.
//!
//! Defines [`FitnessEvaluator`] for computing fitness and [`FitnessComparator`]
//! for determining which fitness value is better. Both traits have blanket
//! implementations for closures, and the module provides [`Maximize`] and
//! [`Minimize`] comparators out of the box.

/// Evaluates the fitness of a genome.
///
/// Implemented automatically for any closure `Fn(&G) -> F`, so you can pass
/// a function directly where a `FitnessEvaluator` is expected.
///
/// # Examples
///
/// ```
/// use evolve::fitness::FitnessEvaluator;
///
/// // A closure works as a fitness evaluator
/// let eval = |g: &[u8; 3]| g.iter().map(|x| *x as u32).sum::<u32>();
/// assert_eq!(eval.evaluate(&[1, 2, 3]), 6);
/// ```
pub trait FitnessEvaluator<G, F> {
    /// Computes the fitness of the given genome.
    fn evaluate(&self, genome: &G) -> F;
}

impl<G, F, E> FitnessEvaluator<G, F> for E
where
    E: Fn(&G) -> F,
{
    fn evaluate(&self, genome: &G) -> F {
        self(genome)
    }
}

/// Compares two fitness values to determine which is better.
///
/// Implemented automatically for any closure `Fn(&F, &F) -> bool`, so you can
/// pass a custom comparison function where a `FitnessComparator` is expected.
///
/// # Examples
///
/// ```
/// use evolve::fitness::FitnessComparator;
///
/// // A closure works as a fitness comparator
/// let cmp = |a: &i32, b: &i32| a > b;
/// assert!(cmp.is_better(&10, &5));
/// ```
pub trait FitnessComparator<F> {
    /// Returns `true` if `f1` is a better fitness than `f2`.
    fn is_better(&self, f1: &F, f2: &F) -> bool;
}

impl<F, C> FitnessComparator<F> for C
where
    C: Fn(&F, &F) -> bool,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        self(f1, f2)
    }
}

/// A [`FitnessComparator`] that treats higher fitness values as better.
///
/// # Examples
///
/// ```
/// use evolve::fitness::{FitnessComparator, Maximize};
///
/// assert!(Maximize.is_better(&10, &5));
/// assert!(!Maximize.is_better(&5, &10));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Maximize;

impl<F> FitnessComparator<F> for Maximize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 > f2
    }
}

/// A [`FitnessComparator`] that treats lower fitness values as better.
///
/// # Examples
///
/// ```
/// use evolve::fitness::{FitnessComparator, Minimize};
///
/// assert!(Minimize.is_better(&5, &10));
/// assert!(!Minimize.is_better(&10, &5));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Minimize;

impl<F> FitnessComparator<F> for Minimize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 < f2
    }
}
