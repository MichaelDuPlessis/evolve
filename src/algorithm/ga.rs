use crate::{
    core::{context::Context, individual::Individual, state::State},
    fitness::{FitnessComparator, FitnessEvaluator, Maximize},
    initialization::Initializer,
    operators::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// The main genetic algorithm runner.
///
/// Wires together an initializer, termination condition, fitness evaluator,
/// genetic operators, and fitness comparator into a runnable algorithm.
///
/// # Examples
///
/// ```
/// use evolve::{
///     algorithm::ga::GeneticAlgorithm,
///     fitness::Maximize,
///     initialization::Random,
///     operators::combinator::Fill,
///     operators::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ga = GeneticAlgorithm::new(
///     Random::new(),
///     MaxGenerations::new(100),
///     |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
///     Fill::from_population_size(RandomReset::new()),
///     NonZero::new(500).unwrap(),
///     rand::rng(),
///     Maximize,
/// );
///
/// let best = ga.run();
/// ```
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
    /// Creates a new `GeneticAlgorithm` with the given components.
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

    /// Runs the algorithm until the termination condition is met and returns the best individual.
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
