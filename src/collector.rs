//! Collectors for gathering results from algorithm runs.
//!
//! The [`Collector`] trait provides hooks that are called at each stage of a
//! algorithm run, and a [`finalize`](Collector::finalize) method to
//! produce a final result.

pub mod basic;
pub mod logger;
pub mod standard;

use crate::core::state::State;
pub use logger::Logger;
pub use standard::RunResult;

/// Collects data during an algorithm run and produces a final result.
///
/// # Examples
///
/// ```
/// use evolve::collector::Collector;
/// use evolve::core::state::State;
///
/// struct GenerationCounter(usize);
///
/// impl<G, F, Fe, C> Collector<G, F, Fe, C> for GenerationCounter {
///     type Result = usize;
///
///     fn on_generation(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {
///         self.0 += 1;
///     }
///
///     fn finalize(self, _state: State<G, F>) -> usize {
///         self.0
///     }
/// }
/// ```
pub trait Collector<G, F, Fe, C> {
    type Result;

    fn on_start(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn on_generation(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn on_end(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {}
    fn finalize(self, state: State<G, F>) -> Self::Result;
}

impl<G, F, Fe, C, T> Collector<G, F, Fe, C> for Box<T>
where
    T: Collector<G, F, Fe, C>,
{
    type Result = T::Result;

    fn on_start(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        (**self).on_start(state, fe, cmp)
    }

    fn on_generation(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        (**self).on_generation(state, fe, cmp)
    }

    fn on_end(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        (**self).on_end(state, fe, cmp)
    }

    fn finalize(self, state: State<G, F>) -> Self::Result {
        (*self).finalize(state)
    }
}

/// A no-op collector that discards the final state and returns `()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOp;

impl<G, F, Fe, C> Collector<G, F, Fe, C> for NoOp {
    type Result = ();

    fn finalize(self, _state: State<G, F>) {}
}

#[cfg(test)]
mod test {
    use crate::algorithm::EvolutionaryAlgorithm;
    use crate::collector::NoOp;
    use crate::fitness::Maximize;
    use crate::initialization::Random;
    use crate::operators::sequential::combinator::Fill;
    use crate::operators::sequential::mutation::RandomReset;
    use crate::termination::MaxGenerations;
    use std::num::NonZero;

    #[test]
    fn noop_returns_unit() {
        let mut ga = EvolutionaryAlgorithm::new(
            Random::new(),
            MaxGenerations::new(10),
            |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
            Fill::from_population_size(RandomReset::new()),
            NonZero::new(20).unwrap(),
            rand::rng(),
            Maximize,
        );
        let result = ga.run_with(NoOp);
        assert_eq!(result, ());
    }
}
