use crate::{
    core::{context::Context, offspring::Offspring, state::State},
    fitness::{FitnessComparator, FitnessEvaluator},
    operators::GeneticOperator,
};
use rand::{Rng, seq::IndexedRandom};
use std::num::NonZero;

/// Tournament selection with replacement.
///
/// Randomly samples `tournament_size` individuals from the population and returns
/// the one with the best fitness. Produces a single [`Offspring::Single`].
///
/// # Examples
///
/// ```
/// use evolve::operators::selection::TournamentSelection;
/// use std::num::NonZero;
///
/// let selection = TournamentSelection::new(NonZero::new(3).unwrap());
/// ```
#[derive(Debug)]
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    /// Creates a new `TournamentSelection` with the given tournament size.
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

        assert!(!population.is_empty());

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
