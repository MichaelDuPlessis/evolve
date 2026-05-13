//! Experiment runner for executing multiple independent trials.

use crate::algorithm::EvolutionaryAlgorithm;
use crate::collector::Collector;
use crate::fitness::FitnessEvaluator;
use crate::initialization::Initializer;
use crate::operators::GeneticOperator;
use crate::termination::TerminationCondition;

/// Creates a fresh instance for each trial in an experiment.
pub trait Factory<A> {
    /// Create a new instance.
    fn create(&mut self) -> A;
}

impl<A, F: FnMut() -> A> Factory<A> for F {
    fn create(&mut self) -> A {
        (self)()
    }
}

/// Runs multiple independent trials of an evolutionary algorithm.
///
/// Construct with [`Experiment::new`], then call [`run`](Experiment::run).
pub struct Experiment<Fac, ColFac> {
    algorithm_factory: Fac,
    trials: usize,
    collector_factory: ColFac,
}

impl<Fac, ColFac> Experiment<Fac, ColFac> {
    /// Creates a new experiment with the given algorithm factory, trial count, and collector factory.
    pub fn new(algorithm_factory: Fac, trials: usize, collector_factory: ColFac) -> Self {
        Self {
            algorithm_factory,
            trials,
            collector_factory,
        }
    }

    /// Runs all trials, returning a result per trial.
    pub fn run<G, F, I, T, Fe, Ops, R, C, Col>(mut self) -> Vec<Col::Result>
    where
        Fac: Factory<EvolutionaryAlgorithm<G, F, I, T, Fe, Ops, R, C>>,
        ColFac: Factory<Col>,
        Col: Collector<G, F, Fe, C>,
        I: Initializer<G, F, Fe, R, C>,
        T: TerminationCondition<G, F>,
        Fe: FitnessEvaluator<G, F>,
        Ops: GeneticOperator<G, F, Fe, R, C>,
    {
        (0..self.trials)
            .map(|_| {
                let mut ea = self.algorithm_factory.create();
                let col = self.collector_factory.create();
                ea.run_with(col)
            })
            .collect()
    }
}
