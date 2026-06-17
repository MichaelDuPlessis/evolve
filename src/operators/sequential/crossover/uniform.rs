use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::{
        GeneticOperator,
        common::{uniform_crossover, uniform_crossover_vec},
    },
};
use rand::Rng;
use std::marker::PhantomData;

/// Uniform crossover operator.
///
/// For each gene position, randomly selects which parent contributes to each child.
/// Each gene has a 50% chance of being swapped between parents.
/// If the population has an odd number of individuals, the last one is dropped.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::crossover::Uniform;
///
/// let crossover = Uniform::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Uniform<T>(PhantomData<T>);

impl<T> Uniform<T> {
    /// Creates a new `Uniform` operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for Uniform<T>
where
    T: Clone,
    R: Rng,
    Fe: FitnessEvaluator<[T; N], F>,
{
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[T; N], F> {
        let mut population = Population::with_capacity(state.population().len());

        for chunk in state.population().chunks_exact(2) {
            let p1 = unsafe { chunk.get_unchecked(0) };
            let p2 = unsafe { chunk.get_unchecked(1) };

            let (child1, child2) = uniform_crossover(p1.genome(), p2.genome(), ctx.rng());

            population.add(Individual::new(child1));
            population.add(Individual::new(child2));
        }

        Offspring::Multiple(population)
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for Uniform<T>
where
    T: Clone,
    R: Rng,
    Fe: FitnessEvaluator<Vec<T>, F>,
{
    fn apply(&self, state: &State<Vec<T>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<T>, F> {
        let mut population = Population::with_capacity(state.population().len());

        for chunk in state.population().chunks_exact(2) {
            let p1 = unsafe { chunk.get_unchecked(0) };
            let p2 = unsafe { chunk.get_unchecked(1) };

            let (child1, child2) = uniform_crossover_vec(p1.genome(), p2.genome(), ctx.rng());

            population.add(Individual::new(child1));
            population.add(Individual::new(child2));
        }

        Offspring::Multiple(population)
    }
}
