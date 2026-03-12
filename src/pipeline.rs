use crate::{
    core::{context::Context, population::Population},
    fitness::FitnessEvaluation,
    operators::GeneticOperator,
};

/// Trait to apply a pipeline of operators
pub trait OperatorPipeline<G, F, Fe> {
    fn apply_operators(&self, ctx: &Context<G, F, Fe>) -> Population<G, F>;
}

/// Base case: single operator
impl<G, F, Fe, O> OperatorPipeline<G, F, Fe> for (O,)
where
    O: GeneticOperator<G, F, Fe>,
    Fe: FitnessEvaluation<G, F>,
{
    fn apply_operators(&self, ctx: &Context<G, F, Fe>) -> Population<G, F> {
        // Apply operator and get the new individuals
        let offspring = self.0.generate(ctx);
        offspring
    }
}

/// Recursive case: operator + rest
impl<G, F, Fe, O, Rest> OperatorPipeline<G, F, Fe> for (O, Rest)
where
    O: GeneticOperator<G, F, Fe>,
    Rest: OperatorPipeline<G, F, Fe>,
    Fe: FitnessEvaluation<G, F>,
{
    fn apply_operators(&self, ctx: &Context<G, F, Fe>) -> Population<G, F> {
        // Generate offspring from the first operator
        let mut new_individuals = self.0.generate(ctx);

        // Generate offspring from the rest of the operators, still referencing original population
        let rest_offspring = self.1.apply_operators(ctx);

        // Append all the rest of the offspring
        new_individuals.extend(rest_offspring.into_vec());
        new_individuals
    }
}
