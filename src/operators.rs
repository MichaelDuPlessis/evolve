pub mod crossover;
pub mod mutation;
pub mod selection;

use crate::{
    core::{
        context::{Context, State},
        population::Population,
    },
    fitness::FitnessEvaluator,
};
use rand::Rng;

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F, Fe, R, C>
where
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F>;

    /// Given an input size this function will return the number of offspring it will produce
    fn output_size(&self, input_size: usize) -> Option<usize>;
}

// Base case: single operator
impl<G, F, Fe, R, O, C> GeneticOperator<G, F, Fe, R, C> for (O,)
where
    O: GeneticOperator<G, F, Fe, R, C>,
    Fe: FitnessEvaluator<G, F>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F> {
        // Apply operator and get the new individuals
        let offspring = self.0.apply(state, ctx);
        offspring
    }

    fn output_size(&self, input_size: usize) -> Option<usize> {
        self.0.output_size(input_size)
    }
}

// Recursive case: operator + rest
impl<G, F, Fe, R, C, O, Rest> GeneticOperator<G, F, Fe, R, C> for (O, Rest)
where
    O: GeneticOperator<G, F, Fe, R, C>,
    Rest: GeneticOperator<G, F, Fe, R, C>,
    Fe: FitnessEvaluator<G, F>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F> {
        let offspring = self.0.apply(state, ctx);

        let next_state = state.with_population(offspring);

        self.1.apply(&next_state, ctx)
    }

    fn output_size(&self, input_size: usize) -> Option<usize> {
        let mid = self.0.output_size(input_size)?;
        self.1.output_size(mid)
    }
}
