use crate::{
    core::{
        context::{Context, State},
        individual::Individual,
        population::Population,
    },
    fitness::FitnessEvaluation,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;

/// Single point crossover operator
pub struct SinglePoint<T>(PhantomData<T>);

impl<T> SinglePoint<T> {
    /// Creates a new `SinglePoint` mutation.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, F, Fe, R, const N: usize> GeneticOperator<[T; N], F, Fe, R> for SinglePoint<T>
where
    T: Clone,
    R: Rng,
    Fe: FitnessEvaluation<[T; N], F>,
{
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R>) -> Population<[T; N], F> {
        let mut population = Population::with_capacity(state.population().len());

        for chunk in state.population().chunks_exact(2) {
            let p1 = unsafe { chunk.get_unchecked(0) };
            let p2 = unsafe { chunk.get_unchecked(1) };

            let point = ctx.rng().random_range(0..N);

            let mut child1 = p1.genome.clone();
            let mut child2 = p2.genome.clone();

            for i in point..N {
                child1[i] = p2.genome[i].clone();
                child2[i] = p1.genome[i].clone();
            }

            let c1 = Individual::new(child1);
            let c2 = Individual::new(child2);

            population.add(c1);
            population.add(c2);
        }

        population
    }
}
