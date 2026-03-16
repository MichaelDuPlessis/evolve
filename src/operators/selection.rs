use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
        population::Population,
    },
    fitness::{FitnessComparator, FitnessEvaluator},
    operators::GeneticOperator,
};
use rand::Rng;

pub struct TournamentSelection {
    tournament_size: usize,
}

impl<G, F, R, Fe, C> GeneticOperator<G, F, Fe, R, C> for TournamentSelection
where
    G: Clone,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
    C: FitnessComparator<F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Population<G, F> {
        todo!()
    }
}
