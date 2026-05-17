use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::{
        GeneticOperator,
        common::{single_point_crossover, single_point_crossover_vec},
    },
};
use rand::Rng;
use std::marker::PhantomData;

/// Single-point crossover operator.
///
/// Pairs individuals from the population, picks a random crossover point, and
/// swaps the genes after that point between each pair to produce two children.
/// If the population has an odd number of individuals, the last one is dropped.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::crossover::SinglePoint;
///
/// let crossover = SinglePoint::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct SinglePoint<T>(PhantomData<T>);

impl<T> SinglePoint<T> {
    /// Creates a new `SinglePoint` crossover operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for SinglePoint<T>
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

            let (child1, child2) = single_point_crossover(p1.genome(), p2.genome(), ctx.rng());

            population.add(Individual::new(child1));
            population.add(Individual::new(child2));
        }

        Offspring::Multiple(population)
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for SinglePoint<T>
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

            let (child1, child2) = single_point_crossover_vec(p1.genome(), p2.genome(), ctx.rng());

            population.add(Individual::new(child1));
            population.add(Individual::new(child2));
        }

        Offspring::Multiple(population)
    }
}
