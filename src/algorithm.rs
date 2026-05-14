//! Algorithm runners.
//!
//! Contains [`EvolutionaryAlgorithm`], which wires together
//! initialization, operators, and termination into a runnable algorithm.

use crate::{
    collector::{Collector, standard},
    core::{context::Context, state::State},
    fitness::{FitnessComparator, FitnessEvaluator, Maximize},
    initialization::Initializer,
    operators::GeneticOperator,
    termination::TerminationCondition,
};
use std::{marker::PhantomData, num::NonZero};

/// The main evolutionary algorithm runner.
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
///     algorithm::EvolutionaryAlgorithm,
///     fitness::Maximize,
///     initialization::Random,
///     operators::sequential::combinator::Fill,
///     operators::sequential::mutation::RandomReset,
///     termination::MaxGenerations,
/// };
/// use std::num::NonZero;
///
/// let mut ga = EvolutionaryAlgorithm::new(
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
pub struct EvolutionaryAlgorithm<G, F, I, T, Fe, Ops, R, C = Maximize> {
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    comparator: C,
    #[cfg(feature = "parallel")]
    runtime: pooled::Runtime,
    _marker: PhantomData<(G, F)>,
}

impl EvolutionaryAlgorithm<(), (), (), (), (), (), (), ()> {
    /// Returns an [`EvolutionaryAlgorithmBuilder`] for incremental construction.
    ///
    /// See [`EvolutionaryAlgorithmBuilder::new`] for examples.
    pub fn builder(
        population_size: NonZero<usize>,
    ) -> EvolutionaryAlgorithmBuilder<(), (), (), (), (), (), (), ()> {
        EvolutionaryAlgorithmBuilder::new(population_size)
    }
}

impl<G, F, I, T, Fe, Ops, R, C> EvolutionaryAlgorithm<G, F, I, T, Fe, Ops, R, C>
where
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
{
    /// Creates a new `EvolutionaryAlgorithm` with the given components.
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
            #[cfg(feature = "parallel")]
            runtime: pooled::Runtime::new(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            ),
            _marker: PhantomData,
        }
    }

    /// Runs the algorithm until the termination condition is met and returns a [`RunResult`](collector::standard::RunResult).
    pub fn run(&mut self) -> standard::RunResult<G, F>
    where
        F: Clone + PartialOrd,
        C: FitnessComparator<F>,
    {
        self.run_with(standard::Standard::default())
    }

    /// Runs the algorithm with a [`Collector`] that is notified at each stage of execution.
    pub fn run_with<Col>(&mut self, mut collector: Col) -> Col::Result
    where
        Col: Collector<G, F, Fe, C>,
    {
        #[cfg(not(feature = "parallel"))]
        let mut ctx = Context::new(&self.fitness_evaluator, &mut self.rng, &self.comparator);
        #[cfg(feature = "parallel")]
        let mut ctx = Context::new(
            &self.fitness_evaluator,
            &mut self.rng,
            &self.comparator,
            &self.runtime,
        );

        let population = self.initializer.initialize(self.population_size, &mut ctx);

        let mut state = State::new(population, 0);

        collector.on_start(&state, ctx.fitness_evaluator(), ctx.comparator());

        while !self.termination.should_terminate(&state) {
            // Apply pipeline — ownership flows through
            state.apply_operators(&mut ctx, &mut self.operators);
            state.inc_generation();

            collector.on_generation(&state, ctx.fitness_evaluator(), ctx.comparator());
        }

        collector.on_end(&state, ctx.fitness_evaluator(), ctx.comparator());

        collector.finalize(state)
    }
}

/// A builder for [`EvolutionaryAlgorithm`] that allows incremental construction.
///
/// All fields must be set before [`build`](Self::build) can be called.
/// The compiler enforces this — `build` is only available when all required
/// types satisfy their trait bounds.
pub struct EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C> {
    initializer: I,
    termination: T,
    fitness_evaluator: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    rng: R,
    comparator: C,
    #[cfg(feature = "parallel")]
    runtime: Option<pooled::Runtime>,
    _marker: PhantomData<(G, F)>,
}

impl EvolutionaryAlgorithmBuilder<(), (), (), (), (), (), (), ()> {
    /// Returns a new [`EvolutionaryAlgorithmBuilder`] for incremental construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use evolve::{
    ///     algorithm::EvolutionaryAlgorithmBuilder,
    ///     fitness::Maximize,
    ///     initialization::Random,
    ///     operators::sequential::combinator::Fill,
    ///     operators::sequential::mutation::RandomReset,
    ///     termination::MaxGenerations,
    /// };
    /// use std::num::NonZero;
    ///
    /// let mut ga = EvolutionaryAlgorithmBuilder::new(NonZero::new(500).unwrap())
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
            #[cfg(feature = "parallel")]
            runtime: None,
            _marker: PhantomData,
        }
    }
}

impl<G, F, I, T, Fe, Ops, R, C> EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C> {
    /// Sets the population initializer.
    pub fn initializer<I2>(
        self,
        initializer: I2,
    ) -> EvolutionaryAlgorithmBuilder<G, F, I2, T, Fe, Ops, R, C> {
        EvolutionaryAlgorithmBuilder {
            initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the termination condition.
    pub fn termination<T2>(
        self,
        termination: T2,
    ) -> EvolutionaryAlgorithmBuilder<G, F, I, T2, Fe, Ops, R, C> {
        EvolutionaryAlgorithmBuilder {
            initializer: self.initializer,
            termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the fitness evaluator.
    pub fn fitness<G2, F2, Fe2>(
        self,
        fitness_evaluator: Fe2,
    ) -> EvolutionaryAlgorithmBuilder<G2, F2, I, T, Fe2, Ops, R, C>
    where
        Fe2: FitnessEvaluator<G2, F2>,
    {
        EvolutionaryAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the genetic operators.
    pub fn operators<Ops2>(
        self,
        operators: Ops2,
    ) -> EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops2, R, C> {
        EvolutionaryAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the random number generator.
    pub fn rng<R2>(self, rng: R2) -> EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R2, C> {
        EvolutionaryAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the fitness comparator.
    pub fn comparator<C2>(
        self,
        comparator: C2,
    ) -> EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C2> {
        EvolutionaryAlgorithmBuilder {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Sets the thread pool runtime for parallel operations.
    #[cfg(feature = "parallel")]
    pub fn runtime(mut self, runtime: pooled::Runtime) -> Self {
        self.runtime = Some(runtime);
        self
    }
}

impl<G, F, I, T, Fe, Ops, R, C> EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C>
where
    I: Initializer<G, F, Fe, R, C>,
    T: TerminationCondition<G, F>,
    Fe: FitnessEvaluator<G, F>,
    Ops: GeneticOperator<G, F, Fe, R, C>,
{
    /// Builds the [`EvolutionaryAlgorithm`].
    ///
    /// Only available when all fields have been set to types that satisfy
    /// their required trait bounds.
    pub fn build(self) -> EvolutionaryAlgorithm<G, F, I, T, Fe, Ops, R, C> {
        EvolutionaryAlgorithm {
            initializer: self.initializer,
            termination: self.termination,
            fitness_evaluator: self.fitness_evaluator,
            operators: self.operators,
            population_size: self.population_size,
            rng: self.rng,
            comparator: self.comparator,
            #[cfg(feature = "parallel")]
            runtime: self.runtime.unwrap_or_else(|| {
                pooled::Runtime::new(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1),
                )
            }),
            _marker: PhantomData,
        }
    }
}

impl<G, F, I, T, Fe, Ops, R, C> std::fmt::Debug for EvolutionaryAlgorithm<G, F, I, T, Fe, Ops, R, C>
where
    I: std::fmt::Debug,
    T: std::fmt::Debug,
    Fe: std::fmt::Debug,
    Ops: std::fmt::Debug,
    R: std::fmt::Debug,
    C: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionaryAlgorithm")
            .field("initializer", &self.initializer)
            .field("termination", &self.termination)
            .field("fitness_evaluator", &self.fitness_evaluator)
            .field("operators", &self.operators)
            .field("population_size", &self.population_size)
            .field("rng", &self.rng)
            .field("comparator", &self.comparator)
            .finish_non_exhaustive()
    }
}

impl<G, F, I, T, Fe, Ops, R, C> std::fmt::Debug
    for EvolutionaryAlgorithmBuilder<G, F, I, T, Fe, Ops, R, C>
where
    I: std::fmt::Debug,
    T: std::fmt::Debug,
    Fe: std::fmt::Debug,
    Ops: std::fmt::Debug,
    R: std::fmt::Debug,
    C: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionaryAlgorithmBuilder")
            .field("initializer", &self.initializer)
            .field("termination", &self.termination)
            .field("fitness_evaluator", &self.fitness_evaluator)
            .field("operators", &self.operators)
            .field("population_size", &self.population_size)
            .field("rng", &self.rng)
            .field("comparator", &self.comparator)
            .finish_non_exhaustive()
    }
}
