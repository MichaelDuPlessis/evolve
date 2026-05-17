use crate::{
    core::{context::Context, offspring::Offspring, state::State},
    fitness::{FitnessComparator, FitnessEvaluator},
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};

/// Rank-based selection.
///
/// Selects a single individual with probability proportional to its rank.
/// The best individual has the highest selection probability, the worst has
/// the lowest. Unlike roulette wheel selection, this avoids dominance by
/// a single high-fitness individual.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::selection::Rank;
///
/// let selection = Rank;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Rank;

impl<G, F, R, Fe, C> GeneticOperator<G, F, Fe, R, C> for Rank
where
    G: Clone,
    F: PartialOrd + Clone,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
    C: FitnessComparator<F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let population = state.population();
        assert!(!population.is_empty());

        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            let fa = population.as_slice()[a].fitness(ctx.fitness_evaluator());
            let fb = population.as_slice()[b].fitness(ctx.fitness_evaluator());
            if ctx.comparator().is_better(fa, fb) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        let n = population.len();
        let total = n * (n + 1) / 2;
        let mut roll = ctx.rng().random_range(0..total);

        for (rank, &idx) in indices.iter().enumerate() {
            let weight = n - rank;
            if roll < weight {
                return Offspring::Single(population.as_slice()[idx].clone_genome_only());
            }
            roll -= weight;
        }

        let last_idx = *indices.last().unwrap();
        Offspring::Single(population.as_slice()[last_idx].clone_genome_only())
    }
}
