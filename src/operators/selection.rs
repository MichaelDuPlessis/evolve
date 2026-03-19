use crate::{
    core::{
        context::{Context, State},
        offspring::Offspring,
    },
    fitness::{FitnessComparator, FitnessEvaluator},
    operators::GeneticOperator,
};
use rand::{Rng, seq::IndexedRandom};
use std::num::NonZero;

/// Performs tournament selection with replacement and returns a single `Individual`.
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: NonZero<usize>) -> Self {
        Self {
            tournament_size: tournament_size.get(),
        }
    }
}

impl<G, F, R, Fe, C> GeneticOperator<G, F, Fe, R, C> for TournamentSelection
where
    G: Clone,
    F: PartialOrd + Clone,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
    C: FitnessComparator<F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let population = state.population();

        assert!(population.len() > 0);

        let mut best = unsafe { population.choose(ctx.rng()).unwrap_unchecked() }; // we already checked to make sure there is at least 1 individual

        for _ in 1..self.tournament_size {
            let candidate = unsafe { population.as_slice().choose(ctx.rng()).unwrap_unchecked() };

            if ctx
                .comparator()
                .is_better(candidate.fitness(), best.fitness())
            {
                best = candidate;
            }
        }

        // Clone the selected individual
        let selected = best.clone();

        Offspring::Single(selected)
    }
}
