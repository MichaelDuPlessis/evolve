use crate::{
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluation,
    initialization::Initialization,
    pipeline::OperatorPipeline,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// Genetic Algorithm runner
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition,
    Fe: FitnessEvaluation<G, F>,
    Ops: OperatorPipeline<G, F, Fe>,
{
    initialization: I,
    termination: T,
    fitness: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    _marker: PhantomData<(G, F)>,
}

impl<G, F, I, T, Fe, Ops> GeneticAlgorithm<G, F, I, T, Fe, Ops>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition,
    Fe: FitnessEvaluation<G, F>,
    Ops: OperatorPipeline<G, F, Fe>,
{
    pub fn new(
        initialization: I,
        termination: T,
        fitness: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
    ) -> Self {
        Self {
            initialization,
            termination,
            fitness,
            operators,
            population_size,
            _marker: PhantomData,
        }
    }

    pub fn run(&self) -> Individual<G, F> {
        let genomes = self.initialization.initialize(self.population_size);
        let mut population = Population::from(genomes);

        let mut generation = 0;
        while !self.termination.should_terminate(generation) {
            // Apply pipeline — ownership flows through
            let ctx = Context::new(&population, &self.fitness, generation);
            population = self.operators.apply_operators(&ctx);
            generation += 1;
        }

        population.best()
    }
}
