use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};

/// Arithmetic crossover for continuous-valued genomes.
///
/// Blends two parents using a random weight α sampled per pair:
/// - child1 = α·parent1 + (1-α)·parent2
/// - child2 = (1-α)·parent1 + α·parent2
///
/// Works with `f64` and `f32` genome types.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::crossover::Arithmetic;
///
/// let crossover = Arithmetic::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Arithmetic;

impl Arithmetic {
    /// Creates a new `Arithmetic` crossover operator.
    pub fn new() -> Self {
        Self
    }
}

macro_rules! impl_arithmetic_crossover {
    ($float:ty, $cast:expr) => {
        impl<F, Fe, R, C, const N: usize> GeneticOperator<[$float; N], F, Fe, R, C> for Arithmetic
        where
            R: Rng,
            Fe: FitnessEvaluator<[$float; N], F>,
        {
            fn apply(
                &self,
                state: &State<[$float; N], F>,
                ctx: &mut Context<Fe, R, C>,
            ) -> Offspring<[$float; N], F> {
                let mut population = Population::with_capacity(state.population().len());

                for chunk in state.population().chunks_exact(2) {
                    let p1 = unsafe { chunk.get_unchecked(0) }.genome();
                    let p2 = unsafe { chunk.get_unchecked(1) }.genome();
                    let alpha: f64 = ctx.rng().random();

                    let child1: [$float; N] = std::array::from_fn(|i| {
                        $cast(alpha * p1[i] as f64 + (1.0 - alpha) * p2[i] as f64)
                    });
                    let child2: [$float; N] = std::array::from_fn(|i| {
                        $cast((1.0 - alpha) * p1[i] as f64 + alpha * p2[i] as f64)
                    });

                    population.add(Individual::new(child1));
                    population.add(Individual::new(child2));
                }

                Offspring::Multiple(population)
            }
        }

        impl<F, Fe, R, C> GeneticOperator<Vec<$float>, F, Fe, R, C> for Arithmetic
        where
            R: Rng,
            Fe: FitnessEvaluator<Vec<$float>, F>,
        {
            fn apply(
                &self,
                state: &State<Vec<$float>, F>,
                ctx: &mut Context<Fe, R, C>,
            ) -> Offspring<Vec<$float>, F> {
                let mut population = Population::with_capacity(state.population().len());

                for chunk in state.population().chunks_exact(2) {
                    let p1 = unsafe { chunk.get_unchecked(0) }.genome();
                    let p2 = unsafe { chunk.get_unchecked(1) }.genome();
                    let alpha: f64 = ctx.rng().random();
                    let len = p1.len().min(p2.len());

                    let child1: Vec<$float> = (0..len)
                        .map(|i| $cast(alpha * p1[i] as f64 + (1.0 - alpha) * p2[i] as f64))
                        .collect();
                    let child2: Vec<$float> = (0..len)
                        .map(|i| $cast((1.0 - alpha) * p1[i] as f64 + alpha * p2[i] as f64))
                        .collect();

                    population.add(Individual::new(child1));
                    population.add(Individual::new(child2));
                }

                Offspring::Multiple(population)
            }
        }
    };
}

impl_arithmetic_crossover!(f64, |x: f64| x);
impl_arithmetic_crossover!(f32, |x: f64| x as f32);
