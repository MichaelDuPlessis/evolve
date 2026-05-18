use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::fitness::FitnessEvaluator;
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// Randomly shuffles a segment of the genome.
///
/// Useful for permutation-based problems where gene values must be preserved
/// but their order matters. Provides more disruption than [`Inversion`](super::Inversion)
/// or [`Swap`](super::Swap).
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::Scramble;
///
/// let mutation = Scramble::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Scramble<T>(PhantomData<T>);

impl<T> Scramble<T> {
    /// Creates a new `Scramble` mutation operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

trait ScrambleCollection {}
impl<T> ScrambleCollection for Vec<T> {}
impl<T> ScrambleCollection for Box<[T]> {}
impl<T> ScrambleCollection for [T] {}
impl<T, const N: usize> ScrambleCollection for [T; N] {}

impl<G, T, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Scramble<T>
where
    G: Clone + AsMut<[T]> + ScrambleCollection,
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
                // Fisher-Yates shuffle on the segment
                for i in (a + 1..=b).rev() {
                    let j = ctx.rng().random_range(a..=i);
                    genes.swap(i, j);
                }
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
                        for i in (a + 1..=b).rev() {
                            let j = ctx.rng().random_range(a..=i);
                            genes.swap(i, j);
                        }
                    }
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}
