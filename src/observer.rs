//! Observers for monitoring algorithm execution.
//!
//! The [`Observer`] trait provides hooks that are called at each stage of a
//! genetic algorithm run. Use it to log progress, collect statistics, or
//! implement custom monitoring.

use crate::{
    core::{context::Context, state::State},
    fitness::FitnessComparator,
};
use std::num::NonZero;

/// A hook into the genetic algorithm's execution lifecycle.
///
/// All methods have default no-op implementations, so you only need to
/// override the ones you care about.
///
/// # Examples
///
/// ```
/// use evolve::{
///     core::{context::Context, state::State},
///     observer::Observer,
/// };
///
/// struct GenerationCounter(usize);
///
/// impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for GenerationCounter {
///     fn on_generation(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
///         self.0 += 1;
///     }
/// }
/// ```
pub trait Observer<G, F, Fe, R, C> {
    /// Called once after the initial population is created, before the main loop.
    fn on_start(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {}

    /// Called at the end of each generation, after operators have been applied.
    fn on_generation(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {}

    /// Called once after the termination condition is met.
    fn on_end(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {}
}

/// A built-in observer that prints the best fitness each generation.
///
/// By default logs every generation. Use [`StatsLogger::new`] to log at a
/// different interval (e.g. every 10th generation).
///
/// Output format: `[gen N] best fitness: F`
#[derive(Debug, Clone, Copy)]
pub struct StatsLogger {
    every: usize,
}

impl StatsLogger {
    /// Creates a `StatsLogger` that logs every `n`th generation.
    pub fn new(n: NonZero<usize>) -> Self {
        Self { every: n.get() }
    }
}

impl Default for StatsLogger {
    fn default() -> Self {
        Self { every: 1 }
    }
}

impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for StatsLogger
where
    F: std::fmt::Display + PartialOrd,
    C: FitnessComparator<F>,
{
    fn on_generation(&mut self, state: &State<G, F>, ctx: &Context<Fe, R, C>) {
        if state.generation() % self.every == 0 {
            let best = state.population().best(ctx.comparator());
            println!(
                "[gen {}] best fitness: {}",
                state.generation(),
                best.fitness()
            );
        }
    }
}

/// An observer that does nothing.
///
/// Used internally by [`GeneticAlgorithm::run`](crate::algorithm::ga::GeneticAlgorithm::run)
/// when no observer is needed.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOp;

impl NoOp {
    /// Creates a new `NoOp` observer.
    pub fn new() -> Self {
        Self
    }
}

impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for NoOp {}

#[cfg(test)]
mod test;
