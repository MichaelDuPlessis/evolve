//! Collectors for gathering results from algorithm runs.
//!
//! The [`Collector`] trait provides hooks that are called at each stage of a
//! genetic algorithm run, and a [`finalize`](Collector::finalize) method to
//! produce a final result.

pub mod basic;
pub mod standard;
pub mod stats_logger;

use crate::core::state::State;
pub use standard::RunResult;

/// Collects data during an algorithm run and produces a final result.
pub trait Collector<G, F, Fe, C> {
    type Result;

    fn on_start(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn on_generation(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn on_end(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn finalize(self, state: State<G, F>) -> Self::Result;
}
