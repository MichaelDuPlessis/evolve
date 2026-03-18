pub mod combinator;
pub mod crossover;
pub mod mutation;
pub mod selection;

use crate::{
    core::{
        context::{Context, State},
        offspring::Offpring,
    },
    fitness::FitnessEvaluator,
};

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F, Fe, R, C>
where
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offpring<G, F>;
}

// // Base case: single operator
// impl<G, F, Fe, R, O, C> GeneticOperator<G, F, Fe, R, C> for (O,)
// where
//     O: GeneticOperator<G, F, Fe, R, C>,
//     Fe: FitnessEvaluator<G, F>,
//     R: Rng,
// {
//     fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F> {
//         // Apply operator and get the new individuals
//         let offspring = self.0.apply(state, ctx);
//         offspring
//     }
// }

// // Recursive case: operator + rest
// impl<G, F, Fe, R, C, O, Rest> GeneticOperator<G, F, Fe, R, C> for (O, Rest)
// where
//     O: GeneticOperator<G, F, Fe, R, C>,
//     Rest: GeneticOperator<G, F, Fe, R, C>,
//     Fe: FitnessEvaluator<G, F>,
//     R: Rng,
// {
//     fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F> {
//         let offspring = self.0.apply(state, ctx);

//         let next_state = state.with_population(offspring);

//         self.1.apply(&next_state, ctx)
//     }
// }
