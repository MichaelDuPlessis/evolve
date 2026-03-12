use crate::core::context::State;

/// Termination condition
pub trait TerminationCondition<G, F> {
    fn should_terminate(&self, state: &State<G, F>) -> bool;
}
