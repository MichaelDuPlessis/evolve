use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::{common::random_reset_mutate, GeneticOperator},
    random::Randomizable,
};
use rand::Rng;
use std::marker::PhantomData;

/// This is a helper trait to get the code to compile since a T may one day implement AsMut<[T]>
trait GeneCollection {}

impl<T> GeneCollection for Vec<T> {}
impl<T> GeneCollection for Box<[T]> {}
impl<T> GeneCollection for [T] {}
impl<T, const N: usize> GeneCollection for [T; N] {}

/// Mutation operator that selects a random gene and replaces it with a new random value.
///
/// When applied to a population with a single individual, returns [`Offspring::Single`].
/// Otherwise returns [`Offspring::Multiple`] with the entire mutated population.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::RandomReset;
///
/// let mutation = RandomReset::<u8>::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct RandomReset<T>(PhantomData<T>);

impl<T> RandomReset<T> {
    /// Creates a new `RandomReset` mutation.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<G, F, R, Fe, T, C> GeneticOperator<G, F, Fe, R, C> for RandomReset<T>
where
    G: Clone + AsMut<[T]> + GeneCollection,
    T: Randomizable<R>,
    F: PartialOrd,
    R: Rng,
    Fe: FitnessEvaluator<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        if state.population().len() == 1 {
            let individual = unsafe { state.population().as_slice().get_unchecked(0) };
            Offspring::Single(Individual::new(random_reset_mutate(individual.genome(), ctx.rng())))
        } else {
            let mut population = Population::with_capacity(state.population().len());

            for individual in state.population() {
                population.add(Individual::new(random_reset_mutate(individual.genome(), ctx.rng())));
            }

            Offspring::Multiple(population)
        }
    }
}

macro_rules! impl_random_reset_for_numbers {
    ($($t:ty),*) => {
        $(
            impl<F, R, Fe, C> GeneticOperator<$t, F, Fe, R, C> for RandomReset<$t>
            where
                F: PartialOrd,
                R: Rng,
                Fe: FitnessEvaluator<$t, F>,
            {
                fn apply(&self, state: &State<$t, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<$t, F> {
                    if state.population().len() == 1 {
                        Offspring::Single(Individual::new(
                            <$t>::random(ctx.rng()),
                        ))
                    } else {
                        let mut population = Population::with_capacity(state.population().len());

                        for _ in state.population() {
                            let new_ind = Individual::new(<$t>::random(ctx.rng()));
                            population.add(new_ind);
                        }

                        Offspring::Multiple(population)
                    }
                }
            }
        )*
    };
}

impl_random_reset_for_numbers!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, char
);
