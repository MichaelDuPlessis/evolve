//! A collector that logs generation statistics to stdout.

use crate::collector::{Collector, NoOp};
use crate::core::state::State;
use crate::fitness::{FitnessComparator, FitnessEvaluator};
use std::num::NonZero;

/// A collector that prints best fitness at a configurable interval.
///
/// Wraps an inner collector, delegating all hooks and adding logging on
/// [`on_generation`](Collector::on_generation).
///
/// # Examples
///
/// ```no_run
/// use evolve::{
///     algorithm::EvolutionaryAlgorithm,
///     collector::Logger,
///     fitness::Maximize,
///     initialization::Random,
///     operators::sequential::combinator::Fill,
///     operators::sequential::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ea = EvolutionaryAlgorithm::new(
///     Random::new(),
///     MaxGenerations::new(100),
///     |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
///     Fill::from_population_size(RandomReset::new()),
///     NonZero::new(50).unwrap(),
///     rand::rng(),
///     Maximize,
/// );
///
/// // Prints "[gen N] best fitness: F" every 10 generations
/// ea.run_with(Logger::new(NonZero::new(10).unwrap()));
/// ```
pub struct Logger<Col = NoOp> {
    every: usize,
    inner: Col,
}

impl Logger {
    /// Create a new `Logger` that logs every `n` generations.
    pub fn new(n: NonZero<usize>) -> Self {
        Self {
            every: n.get(),
            inner: NoOp,
        }
    }
}

impl<Col> Logger<Col> {
    /// Create a new `Logger` that logs every `n` generations, wrapping the given collector.
    pub fn with_collector(n: NonZero<usize>, collector: Col) -> Self {
        Self {
            every: n.get(),
            inner: collector,
        }
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self {
            every: 1,
            inner: NoOp,
        }
    }
}

impl<G, F, Fe, C, Col> Collector<G, F, Fe, C> for Logger<Col>
where
    F: std::fmt::Display + PartialOrd + Clone,
    Fe: FitnessEvaluator<G, F>,
    C: FitnessComparator<F>,
    Col: Collector<G, F, Fe, C>,
{
    type Result = Col::Result;

    fn on_start(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        self.inner.on_start(state, fe, cmp);
    }

    fn on_generation(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        self.inner.on_generation(state, fe, cmp);
        let generation = state.generation();
        if generation.is_multiple_of(self.every) {
            let best = state.population().best(fe, cmp);
            println!("[gen {}] best fitness: {}", generation, best.fitness(fe));
        }
    }

    fn on_end(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        self.inner.on_end(state, fe, cmp);
    }

    fn finalize(self, state: State<G, F>) -> Self::Result {
        self.inner.finalize(state)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::EvolutionaryAlgorithm;
    use crate::collector::basic::Basic;
    use crate::collector::standard::Standard;
    use crate::fitness::Maximize;
    use crate::initialization::Random;
    use crate::operators::sequential::combinator::Fill;
    use crate::operators::sequential::mutation::RandomReset;
    use crate::termination::MaxGenerations;
    use std::num::NonZero;

    #[test]
    fn delegates_to_inner() {
        let mut ga = EvolutionaryAlgorithm::new(
            Random::new(),
            MaxGenerations::new(10),
            |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
            Fill::from_population_size(RandomReset::new()),
            NonZero::new(20).unwrap(),
            rand::rng(),
            Maximize,
        );
        let result = ga.run_with(Logger::with_collector(
            NonZero::new(1).unwrap(),
            Standard::default(),
        ));
        assert_eq!(result.generations(), 10);
        assert!(!result.best_fitness().is_empty());
    }

    #[test]
    fn every_n_skips_generations() {
        let mut ga = EvolutionaryAlgorithm::new(
            Random::new(),
            MaxGenerations::new(10),
            |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
            Fill::from_population_size(RandomReset::new()),
            NonZero::new(20).unwrap(),
            rand::rng(),
            Maximize,
        );
        let result = ga.run_with(Logger::with_collector(
            NonZero::new(5).unwrap(),
            Basic::new(),
        ));
        assert_eq!(result.generations(), 10);
    }
}
