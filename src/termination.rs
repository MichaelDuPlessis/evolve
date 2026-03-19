use crate::core::state::State;

/// Termination condition
pub trait TerminationCondition<G, F> {
    fn should_terminate(&self, state: &State<G, F>) -> bool;
}

/// Terminates the algorithm after a set number of generations.
#[derive(Debug)]
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
