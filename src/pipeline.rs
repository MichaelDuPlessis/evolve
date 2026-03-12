use crate::{core::population::Population, fitness::FitnessEvaluation, operators::GeneticOperator};

/// Trait to apply a pipeline of operators
pub trait OperatorPipeline<G, F> {
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F>;
}

/// Base case: single operator
impl<G, F, O> OperatorPipeline<G, F> for (O,)
where
    O: GeneticOperator<G, F>,
{
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F> {
        self.0.apply(population, fitness)
    }
}

/// Recursive case: operator + rest
impl<G, F, O, Rest> OperatorPipeline<G, F> for (O, Rest)
where
    O: GeneticOperator<G, F>,
    Rest: OperatorPipeline<G, F>,
{
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F> {
        let intermediate = self.0.apply(population, fitness);
        self.1.apply_operators(intermediate, fitness)
    }
}
