pub mod mutation;

use rand::Rng;

use crate::{
    core::{
        context::{Context, State},
        population::Population,
    },
    fitness::FitnessEvaluation,
};

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F, Fe, R>
where
    Fe: FitnessEvaluation<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &Context<Fe, R>) -> Population<G, F>;
}

// Base case: single operator
impl<G, F, Fe, R, O> GeneticOperator<G, F, Fe, R> for (O,)
where
    O: GeneticOperator<G, F, Fe, R>,
    Fe: FitnessEvaluation<G, F>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &Context<Fe, R>) -> Population<G, F> {
        // Apply operator and get the new individuals
        let offspring = self.0.apply(state, ctx);
        offspring
    }
}

// Recursive case: operator + rest
impl<G, F, Fe, R, O, Rest> GeneticOperator<G, F, Fe, R> for (O, Rest)
where
    O: GeneticOperator<G, F, Fe, R>,
    Rest: GeneticOperator<G, F, Fe, R>,
    Fe: FitnessEvaluation<G, F>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &Context<Fe, R>) -> Population<G, F> {
        // Generate offspring from the first operator
        let mut new_individuals = self.0.apply(state, ctx);

        // Generate offspring from the rest of the operators, still referencing original population
        let rest_offspring = self.1.apply(state, ctx);

        // Append all the rest of the offspring
        new_individuals.extend(rest_offspring);
        new_individuals
    }
}
