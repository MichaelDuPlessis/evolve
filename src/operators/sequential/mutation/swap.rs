use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// Swaps two random genes in the genome.
///
/// Useful for permutation-based problems where gene values must be preserved
/// but their order matters (e.g., TSP, scheduling).
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::Swap;
///
/// let mutation = Swap::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Swap<T>(PhantomData<T>);

impl<T> Swap<T> {
    /// Creates a new `Swap` mutation operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

trait SwapCollection {}
impl<T> SwapCollection for Vec<T> {}
impl<T> SwapCollection for Box<[T]> {}
impl<T> SwapCollection for [T] {}
impl<T, const N: usize> SwapCollection for [T; N] {}

impl<G, T, F, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Swap<T>
where
    G: Clone + AsMut<[T]> + SwapCollection,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = individual.genome().clone();
            let genes = genome.as_mut();
            if genes.len() >= 2 {
                let a = ctx.rng().random_range(0..genes.len());
                let b = ctx.rng().random_range(0..genes.len());
                genes.swap(a, b);
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
                        let a = ctx.rng().random_range(0..genes.len());
                        let b = ctx.rng().random_range(0..genes.len());
                        genes.swap(a, b);
                    }
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}
