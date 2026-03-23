//! Termination conditions for the algorithm.
//!
//! Defines the [`TerminationCondition`] trait and provides [`MaxGenerations`],
//! which stops the algorithm after a set number of generations.

use crate::core::state::State;

/// Determines when the algorithm should stop running.
///
/// # Examples
///
/// ```
/// use evolve::core::{population::Population, state::State};
/// use evolve::termination::{MaxGenerations, TerminationCondition};
///
/// let tc = MaxGenerations::new(100);
/// let state = State::<(), ()>::new(Population::new(), 99);
/// assert!(!tc.should_terminate(&state));
///
/// let state = State::<(), ()>::new(Population::new(), 100);
/// assert!(tc.should_terminate(&state));
/// ```
pub trait TerminationCondition<G, F> {
    /// Returns `true` if the algorithm should stop.
    fn should_terminate(&self, state: &State<G, F>) -> bool;
}

/// Terminates the algorithm once the generation count reaches the given limit.
#[derive(Debug, Clone, Copy)]
pub struct MaxGenerations(usize);

impl MaxGenerations {
    /// Creates a new `MaxGenerations` struct.
    pub fn new(max_generations: usize) -> Self {
        Self(max_generations)
    }
}

impl<G, F> TerminationCondition<G, F> for MaxGenerations {
    fn should_terminate(&self, state: &State<G, F>) -> bool {
        state.generation() >= self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::{population::Population, state::State};

    #[test]
    fn terminates_at_max_generation() {
        let tc = MaxGenerations::new(10);
        let state = State::<(), ()>::new(Population::new(), 10);
        assert!(tc.should_terminate(&state));
    }

    #[test]
    fn terminates_past_max_generation() {
        let tc = MaxGenerations::new(10);
        let state = State::<(), ()>::new(Population::new(), 15);
        assert!(tc.should_terminate(&state));
    }

    #[test]
    fn does_not_terminate_before_max() {
        let tc = MaxGenerations::new(10);
        let state = State::<(), ()>::new(Population::new(), 9);
        assert!(!tc.should_terminate(&state));
    }
}
