use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::fitness::FitnessEvaluator;
use crate::operators::GeneticOperator;
use rand::{Rng, RngExt, SeedableRng};

/// Parallel version of [`Gaussian`](crate::operators::sequential::mutation::Gaussian).
///
/// Distributes individuals across pool workers for Gaussian mutation.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    std_dev: f64,
}

impl Gaussian {
    /// Creates a new parallel `Gaussian` mutation with the given standard deviation.
    pub fn new(std_dev: f64) -> Self {
        Self { std_dev }
    }
}

/// Generates a standard normal random value using the Box-Muller transform.
fn box_muller(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

macro_rules! impl_parallel_gaussian {
    ($float:ty, $cast:expr) => {
        impl<F, Fe, R, C, const N: usize> GeneticOperator<[$float; N], F, Fe, R, C> for Gaussian
        where
            $float: Send + Sync,
            F: Send,
            R: Rng + SeedableRng,
            Fe: FitnessEvaluator<[$float; N], F> + Sync,
            C: Sync,
            Individual<[$float; N], F>: Sync,
        {
            fn apply(&self, state: &State<[$float; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[$float; N], F> {
                let individuals = state.population().as_slice();
                let std_dev = self.std_dev;
                let inputs: vecpool::PoolVec<(u64, usize)> = (0..individuals.len())
                    .map(|i| (ctx.rng().random::<u64>(), i))
                    .collect();

                let results = ctx.pool().map(&inputs, |(seed, idx)| {
                    let mut rng = R::seed_from_u64(*seed);
                    let mut genome = *individuals[*idx].genome();
                    let gene_idx = rng.random_range(0..N);
                    genome[gene_idx] += $cast(box_muller(&mut rng) * std_dev);
                    Individual::new(genome)
                });

                let mut population = Population::with_capacity(individuals.len());
                for r in results {
                    population.add(r.expect("pool task panicked"));
                }
                Offspring::Multiple(population)
            }
        }

        impl<F, Fe, R, C> GeneticOperator<Vec<$float>, F, Fe, R, C> for Gaussian
        where
            $float: Send + Sync,
            F: Send,
            R: Rng + SeedableRng,
            Fe: FitnessEvaluator<Vec<$float>, F> + Sync,
            C: Sync,
            Individual<Vec<$float>, F>: Sync,
        {
            fn apply(&self, state: &State<Vec<$float>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<$float>, F> {
                let individuals = state.population().as_slice();
                let std_dev = self.std_dev;
                let inputs: vecpool::PoolVec<(u64, usize)> = (0..individuals.len())
                    .map(|i| (ctx.rng().random::<u64>(), i))
                    .collect();

                let results = ctx.pool().map(&inputs, |(seed, idx)| {
                    let mut rng = R::seed_from_u64(*seed);
                    let mut genome = individuals[*idx].genome().clone();
                    let gene_idx = rng.random_range(0..genome.len());
                    genome[gene_idx] += $cast(box_muller(&mut rng) * std_dev);
                    Individual::new(genome)
                });

                let mut population = Population::with_capacity(individuals.len());
                for r in results {
                    population.add(r.expect("pool task panicked"));
                }
                Offspring::Multiple(population)
            }
        }
    };
}

impl_parallel_gaussian!(f64, |x: f64| x);
impl_parallel_gaussian!(f32, |x: f64| x as f32);
