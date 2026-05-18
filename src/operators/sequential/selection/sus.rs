use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};
use std::num::NonZero;

/// Stochastic Universal Sampling (SUS) selection.
///
/// Selects multiple individuals using evenly spaced pointers on a
/// fitness-proportionate wheel. Produces lower variance than repeated
/// roulette wheel selection.
///
/// Fitness values must be non-negative for meaningful results.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::selection::Sus;
/// use std::num::NonZero;
///
/// // Select 10 individuals per call
/// let selection = Sus::new(NonZero::new(10).unwrap());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Sus {
    count: usize,
}

impl Sus {
    /// Creates a new `Sus` operator that selects `count` individuals per call.
    pub fn new(count: NonZero<usize>) -> Self {
        Self { count: count.get() }
    }
}

impl<G, F, R, Fe, C> GeneticOperator<G, F, Fe, R, C> for Sus
where
    G: Clone,
    F: Copy + Into<f64>,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let population = state.population();
        assert!(!population.is_empty());

        let individuals = population.as_slice();
        let fitnesses: Vec<f64> = individuals
            .iter()
            .map(|ind| (*ind.fitness(ctx.fitness_evaluator())).into())
            .collect();

        let total: f64 = fitnesses.iter().sum();
        let step = total / self.count as f64;
        let start: f64 = ctx.rng().random_range(0.0..step);

        let mut population_out = Population::with_capacity(self.count);
        let mut cumulative = 0.0;
        let mut idx = 0;

        for i in 0..self.count {
            let pointer = start + i as f64 * step;
            while idx < fitnesses.len() - 1 && cumulative + fitnesses[idx] < pointer {
                cumulative += fitnesses[idx];
                idx += 1;
            }
            population_out.add(individuals[idx].clone_genome_only());
        }

        Offspring::Multiple(population_out)
    }
}
