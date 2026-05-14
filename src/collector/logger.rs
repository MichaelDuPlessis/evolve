//! A collector that logs generation statistics to stdout.

use crate::collector::{Collector, NoOp};
use crate::core::state::State;
use crate::fitness::{FitnessComparator, FitnessEvaluator};
use std::num::NonZero;

/// A collector that prints best fitness at a configurable interval.
///
/// Wraps an inner collector, delegating all hooks and adding logging on
/// [`on_generation`](Collector::on_generation).
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
