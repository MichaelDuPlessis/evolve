use std::num::NonZero;

/// Trait to initialize the population
pub trait Initialization<G> {
    fn initialize(&self, population_size: NonZero<usize>) -> Vec<G>;
}
