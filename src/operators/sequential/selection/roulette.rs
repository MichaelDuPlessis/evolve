use crate::{
    core::{context::Context, offspring::Offspring, state::State},
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};

/// Fitness-proportionate (roulette wheel) selection.
///
/// Selects a single individual with probability proportional to its fitness.
/// Higher fitness values have a greater chance of being selected.
///
/// Fitness values must be non-negative for meaningful results.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::selection::RouletteWheel;
///
/// let selection = RouletteWheel;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct RouletteWheel;

impl<G, F, R, Fe, C> GeneticOperator<G, F, Fe, R, C> for RouletteWheel
where
    G: Clone,
    F: Copy + Into<f64>,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let population = state.population();
        assert!(!population.is_empty());

        let total: f64 = population
            .iter()
            .map(|ind| (*ind.fitness(ctx.fitness_evaluator())).into())
            .sum();

        let roll: f64 = ctx.rng().random::<f64>() * total;
        let mut cumulative = 0.0;

        for ind in population {
            cumulative += (*ind.fitness(ctx.fitness_evaluator())).into();
            if cumulative > roll {
                return Offspring::Single(ind.clone_genome_only());
            }
        }

        // Fallback for floating-point rounding edge case
        let last = population.iter().last().unwrap();
        Offspring::Single(last.clone_genome_only())
    }
}
