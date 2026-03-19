pub mod combinator;
pub mod crossover;
pub mod mutation;
pub mod selection;

use crate::core::{context::Context, offspring::Offspring, state::State};

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F, Fe, R, C> {
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F>;
}
