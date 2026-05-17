use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};

/// Gaussian mutation operator for continuous-valued genomes.
///
/// Selects a random gene and adds Gaussian noise with the configured
/// standard deviation. Works with `f64` and `f32` genome types.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::mutation::Gaussian;
///
/// let mutation = Gaussian::new(0.1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    std_dev: f64,
}

impl Gaussian {
    /// Creates a new `Gaussian` with the given standard deviation.
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

impl<F, Fe, R, C, const N: usize> GeneticOperator<[f64; N], F, Fe, R, C> for Gaussian
where
    R: Rng,
    Fe: FitnessEvaluator<[f64; N], F>,
{
    fn apply(
        &self,
        state: &State<[f64; N], F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<[f64; N], F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = *individual.genome();
            let idx = ctx.rng().random_range(0..N);
            genome[idx] += box_muller(ctx.rng()) * self.std_dev;
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(
        &self,
        state: State<[f64; N], F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<[f64; N], F> {
        let std_dev = self.std_dev;
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let idx = ctx.rng().random_range(0..N);
                    genome[idx] += box_muller(ctx.rng()) * std_dev;
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}

impl<F, Fe, R, C> GeneticOperator<Vec<f64>, F, Fe, R, C> for Gaussian
where
    R: Rng,
    Fe: FitnessEvaluator<Vec<f64>, F>,
{
    fn apply(
        &self,
        state: &State<Vec<f64>, F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<Vec<f64>, F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = individual.genome().clone();
            let idx = ctx.rng().random_range(0..genome.len());
            genome[idx] += box_muller(ctx.rng()) * self.std_dev;
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(
        &self,
        state: State<Vec<f64>, F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<Vec<f64>, F> {
        let std_dev = self.std_dev;
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let idx = ctx.rng().random_range(0..genome.len());
                    genome[idx] += box_muller(ctx.rng()) * std_dev;
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}

impl<F, Fe, R, C, const N: usize> GeneticOperator<[f32; N], F, Fe, R, C> for Gaussian
where
    R: Rng,
    Fe: FitnessEvaluator<[f32; N], F>,
{
    fn apply(
        &self,
        state: &State<[f32; N], F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<[f32; N], F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = *individual.genome();
            let idx = ctx.rng().random_range(0..N);
            genome[idx] += (box_muller(ctx.rng()) * self.std_dev) as f32;
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(
        &self,
        state: State<[f32; N], F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<[f32; N], F> {
        let std_dev = self.std_dev;
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let idx = ctx.rng().random_range(0..N);
                    genome[idx] += (box_muller(ctx.rng()) * std_dev) as f32;
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}

impl<F, Fe, R, C> GeneticOperator<Vec<f32>, F, Fe, R, C> for Gaussian
where
    R: Rng,
    Fe: FitnessEvaluator<Vec<f32>, F>,
{
    fn apply(
        &self,
        state: &State<Vec<f32>, F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<Vec<f32>, F> {
        let mut population = Population::with_capacity(state.population().len());
        for individual in state.population() {
            let mut genome = individual.genome().clone();
            let idx = ctx.rng().random_range(0..genome.len());
            genome[idx] += (box_muller(ctx.rng()) * self.std_dev) as f32;
            population.add(Individual::new(genome));
        }
        Offspring::Multiple(population)
    }

    fn transform(
        &self,
        state: State<Vec<f32>, F>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Offspring<Vec<f32>, F> {
        let std_dev = self.std_dev;
        let population = state
            .into_population()
            .into_iter()
            .map(|ind| {
                ind.mutate_genome(|genome| {
                    let idx = ctx.rng().random_range(0..genome.len());
                    genome[idx] += (box_muller(ctx.rng()) * std_dev) as f32;
                })
            })
            .collect();
        Offspring::Multiple(population)
    }
}
