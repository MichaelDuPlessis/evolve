use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
    },
    fitness::{FitnessComparator, FitnessEvaluator, Maximize},
    initialization::Initializer,
    operators::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// Genetic Algorithm runner
#[derive(Debug)]
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops, R, C = Maximize>
where
    F: PartialOrd,
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
{
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    goal: C,
    _marker: PhantomData<(G, F)>,
}

impl<G, F, I, T, Fe, Ops, R, C> GeneticAlgorithm<G, F, I, T, Fe, Ops, R, C>
where
    F: PartialOrd,
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
    C: FitnessComparator<F>,
    // hmm I don't know if I like this
    G: Clone,
    F: Clone,
{
    pub fn new(
        initialization: I,
        termination: T,
        fitness: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
        rng: R,
        goal: C,
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
        let mut ctx = Context::new(&self.fitness_evaluator, &mut self.rng, &self.goal);

        let population = self.initializer.initialize(self.population_size, &mut ctx);

        let mut state = State::new(population, 0);

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.apply_operators(&mut ctx, &mut self.operators);
            state.inc_generation();
        }

        state.into_population().best(&self.goal).clone()
    }
}
