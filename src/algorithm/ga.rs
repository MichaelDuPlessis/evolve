use crate::{
    core::{context::Context, run_result::RunResult, state::State},
    fitness::{FitnessEvaluator, Maximize},
    initialization::Initializer,
    observer::{NoOp, Observer},
    operators::sequential::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// The main genetic algorithm runner.
///
/// Wires together an initializer, termination condition, fitness evaluator,
/// genetic operators, and fitness comparator into a runnable algorithm.
///
/// Can be constructed directly with [`new`](Self::new) or incrementally with
/// [`builder`](Self::builder).
///
/// # Examples
///
/// ```
/// use evolve::{
///     algorithm::ga::GeneticAlgorithm,
///     fitness::Maximize,
///     initialization::Random,
///     operators::sequential::combinator::Fill,
///     operators::sequential::mutation::RandomReset,
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
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops, R, C = Maximize> {
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    comparator: C,
    _marker: PhantomData<(G, F)>,
}

impl GeneticAlgorithm<(), (), (), (), (), (), (), ()> {
    /// Returns a [`GeneticAlgorithmBuilder`] for incremental construction.
    ///
    /// See [`GeneticAlgorithmBuilder::new`] for examples.
    pub fn builder(
        population_size: NonZero<usize>,
    ) -> GeneticAlgorithmBuilder<(), (), (), (), (), (), (), ()> {
        GeneticAlgorithmBuilder::new(population_size)
    }
}

impl<G, F, I, T, Fe, Ops, R, C> GeneticAlgorithm<G, F, I, T, Fe, Ops, R, C>
where
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
{
    /// Creates a new `GeneticAlgorithm` with the given components.
    pub fn new(
        initializer: I,
        termination: T,
        fitness_evaluator: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
        rng: R,
        comparator: C,
    ) -> Self {
        Self {
            initializer,
            termination,
            fitness_evaluator,
            operators,
            population_size,
            rng,
            comparator,
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

        observer.on_start(&state, &ctx);

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.apply_operators(&mut ctx, &mut self.operators);
            state.inc_generation();

            observer.on_generation(&state, &ctx);
        }

        observer.on_end(&state, &ctx);

        state.into()
    }
}

/// A builder for [`GeneticAlgorithm`] that allows incremental construction.
///
/// All fields must be set before [`build`](Self::build) can be called.
/// The compiler enforces this — `build` is only available when all required
/// types satisfy their trait bounds.
pub struct GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C> {
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    comparator: C,
    _marker: PhantomData<(G, F)>,
}

impl GeneticAlgorithmBuilder<(), (), (), (), (), (), (), ()> {
    /// Returns a new [`GeneticAlgorithmBuilder`] for incremental construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::{
    ///     algorithm::ga::GeneticAlgorithmBuilder,
    ///     fitness::Maximize,
    ///     initialization::Random,
    ///     operators::sequential::combinator::Fill,
    ///     operators::sequential::mutation::RandomReset,
    ///     termination::MaxGenerations,
    /// };
    /// use std::num::NonZero;
    ///
    /// let mut ga = GeneticAlgorithmBuilder::new(NonZero::new(500).unwrap())
    ///     .initializer(Random::new())
    ///     .termination(MaxGenerations::new(100))
    ///     .fitness(|g: &[u8; 2]| g[0] as u16 + g[1] as u16)
    ///     .operators(Fill::from_population_size(RandomReset::new()))
    ///     .rng(rand::rng())
    ///     .comparator(Maximize)
    ///     .build();
    ///
    /// let result = ga.run();
    /// ```
    pub fn new(population_size: NonZero<usize>) -> Self {
        Self {
            initializer: (),
            termination: (),
            fitness_evaluator: (),
            operators: (),
            population_size,
            rng: (),
            comparator: (),
            _marker: PhantomData,
        }
    }
}

impl<G, F, I, T, Fe, Ops, R, C> GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C> {
    /// Sets the population initializer.
    pub fn initializer<I2>(
        self,
        initializer: I2,
    ) -> GeneticAlgorithmBuilder<G, F, I2, T, Fe, Ops, R, C> {
        GeneticAlgorithmBuilder {
            initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            _marker: PhantomData,
        }
    }

    /// Sets the termination condition.
    pub fn termination<T2>(
        self,
        termination: T2,
    ) -> GeneticAlgorithmBuilder<G, F, I, T2, Fe, Ops, R, C> {
        GeneticAlgorithmBuilder {
            initializer: self.initializer,
            termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            _marker: PhantomData,
        }
    }

    /// Sets the fitness evaluator.
    pub fn fitness<G2, F2, Fe2>(
        self,
        fitness_evaluator: Fe2,
    ) -> GeneticAlgorithmBuilder<G2, F2, I, T, Fe2, Ops, R, C>
    where
        Fe2: FitnessEvaluator<G2, F2>,
    {
        GeneticAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            _marker: PhantomData,
        }
    }

    /// Sets the genetic operators.
    pub fn operators<Ops2>(
        self,
        operators: Ops2,
    ) -> GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops2, R, C> {
        GeneticAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            _marker: PhantomData,
        }
    }

    /// Sets the random number generator.
    pub fn rng<R2>(self, rng: R2) -> GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops, R2, C> {
        GeneticAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng,
            comparator: self.comparator,
            _marker: PhantomData,
        }
    }

    /// Sets the fitness comparator.
    pub fn comparator<C2>(
        self,
        comparator: C2,
    ) -> GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C2> {
        GeneticAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator,
            _marker: PhantomData,
        }
    }
}

impl<G, F, I, T, Fe, Ops, R, C> GeneticAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C>
where
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
{
    /// Builds the [`GeneticAlgorithm`].
    ///
    /// Only available when all fields have been set to types that satisfy
    /// their required trait bounds.
    pub fn build(self) -> GeneticAlgorithm<G, F, I, T, Fe, Ops, R, C> {
        GeneticAlgorithm::new(
            self.initializer,
            self.termination,
            self.fitness_evaluator,
            self.operators,
            self.population_size,
            self.rng,
            self.comparator,
        )
    }
}
