use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::fitness::FitnessEvaluator;
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// Reverses a random segment of the genome.
///
/// Useful for permutation-based problems where gene values must be preserved
/// but their relative order matters (e.g., TSP, scheduling).
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::Inversion;
///
/// let mutation = Inversion::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Inversion<T>(PhantomData<T>);

impl<T> Inversion<T> {
    /// Creates a new `Inversion` mutation operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

trait InversionCollection {}
impl<T> InversionCollection for Vec<T> {}
impl<T> InversionCollection for Box<[T]> {}
impl<T> InversionCollection for [T] {}
impl<T, const N: usize> InversionCollection for [T; N] {}

impl<G, T, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Inversion<T>
where
    G: Clone + AsMut<[T]> + InversionCollection,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = individual.genome().clone();
            let genes = genome.as_mut();
            if genes.len() >= 2 {
                let mut a = ctx.rng().random_range(0..genes.len());
                let mut b = ctx.rng().random_range(0..genes.len());
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                genes[a..=b].reverse();
            }
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let genes = genome.as_mut();
                    if genes.len() >= 2 {
                        let mut a = ctx.rng().random_range(0..genes.len());
                        let mut b = ctx.rng().random_range(0..genes.len());
                        if a > b {
                            std::mem::swap(&mut a, &mut b);
                        }
                        genes[a..=b].reverse();
                    }
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}
