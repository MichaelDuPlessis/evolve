use crate::{
    core::{context::Context, run_result::RunResult, state::State},
    fitness::{FitnessComparator, FitnessEvaluator, Maximize},
    initialization::Initializer,
    observer::{NoOp, Observer},
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
/// let result = ga.run();
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
    comparator: C,
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
            comparator: goal,
            _marker: PhantomData,
        }
    }

    /// Runs the algorithm until the termination condition is met and returns a [`RunResult`].
    pub fn run(&mut self) -> RunResult<G, F> {
        self.run_with(NoOp::new())
    }

    /// Runs the algorithm with an [`Observer`] that is notified at each stage of execution.
    pub fn run_with<O>(&mut self, mut observer: O) -> RunResult<G, F>
    where
        O: Observer<G, F, Fe, R, C>,
    {
        let mut ctx = Context::new(&self.fitness_evaluator, &mut self.rng, &self.comparator);

        let population = self.initializer.initialize(self.population_size, &mut ctx);

        let mut state = State::new(population, 0);

        observer.on_start(&state, &mut ctx);

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.apply_operators(&mut ctx, &mut self.operators);
            state.inc_generation();

            observer.on_generation(&state, &mut ctx);
        }

        observer.on_end(&state, &mut ctx);

        state.into()
    }
}
