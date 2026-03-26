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

/// Single-point crossover operator.
///
/// Pairs individuals from the population, picks a random crossover point, and
/// swaps the genes after that point between each pair to produce two children.
/// If the population has an odd number of individuals, the last one is dropped.
///
/// # Examples
///
/// ```
/// use evolve::operators::crossover::SinglePoint;
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

            let point = ctx.rng().random_range(0..N);

            let mut child1 = p1.genome().clone();
            let mut child2 = p2.genome().clone();

            child1[point..N].clone_from_slice(&p2.genome()[point..N]);
            child2[point..N].clone_from_slice(&p1.genome()[point..N]);

            let c1 = Individual::new(child1);
            let c2 = Individual::new(child2);

            population.add(c1);
            population.add(c2);
        }

        Offspring::Multiple(population)
    }
}
