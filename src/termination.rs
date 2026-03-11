/// Termination condition
pub trait TerminationCondition {
    fn should_terminate(&self, generation: usize) -> bool;
}
