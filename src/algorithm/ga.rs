use crate::{
    core::{
        context::{Context, State},
        goal::Goal,
        individual::Individual,
    },
    fitness::FitnessEvaluator,
    initialization::Initializer,
    operators::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// Genetic Algorithm runner
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops, R>
where
    F: PartialOrd,
    I: Initializer<G, F, Fe, R>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R>,
{
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    goal: Goal,
    _marker: PhantomData<(G, F)>,
}

impl<G, F, I, T, Fe, Ops, R> GeneticAlgorithm<G, F, I, T, Fe, Ops, R>
where
    F: PartialOrd,
    I: Initializer<G, F, Fe, R>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R>,
{
    pub fn new(
        initialization: I,
        termination: T,
        fitness: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
        rng: R,
        goal: Goal,
    ) -> Self {
        Self {
            initializer: initialization,
            termination,
            fitness_evaluator: fitness,
            operators,
            population_size,
            rng,
            goal,
            _marker: PhantomData,
        }
    }

    pub fn run(&mut self) -> Individual<G, F> {
        let mut ctx = Context::new(&self.fitness_evaluator, &mut self.rng, self.goal);

        let population = self.initializer.initialize(self.population_size, &mut ctx);

        let mut state = State::new(population, 0);

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.apply_operators(&mut ctx, &mut self.operators);
            state.inc_generation();
        }

        todo!()
    }
}
