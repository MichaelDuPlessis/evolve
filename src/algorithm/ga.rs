use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
        population::Population,
    },
    fitness::FitnessEvaluation,
    initialization::Initialization,
    operators::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// Genetic Algorithm runner
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops, R>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluation<G, F>,
    Ops: GeneticOperator<G, F, Fe, R>,
{
    initialization: I,
    termination: T,
    fitness: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    _marker: PhantomData<(G, F)>,
}

impl<G, F, I, T, Fe, Ops, R> GeneticAlgorithm<G, F, I, T, Fe, Ops, R>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluation<G, F>,
    Ops: GeneticOperator<G, F, Fe, R>,
{
    pub fn new(
        initialization: I,
        termination: T,
        fitness: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
        rng: R,
    ) -> Self {
        Self {
            initialization,
            termination,
            fitness,
            operators,
            population_size,
            rng,
            _marker: PhantomData,
        }
    }

    pub fn run(&mut self) -> Individual<G, F> {
        let genomes = self.initialization.initialize(self.population_size);
        let population = Population::from(genomes);

        let mut ctx = Context::new(&self.fitness, &mut self.rng);
        let mut state = State::new(population, 0);

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.population = self.operators.apply(&state, &mut ctx);
            state.generation += 1;
        }

        state.population.best()
    }
}
