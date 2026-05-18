use crate::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{
        GeneticOperator,
        common::{
            single_point_crossover, single_point_crossover_vec, two_point_crossover,
            two_point_crossover_vec, uniform_crossover, uniform_crossover_vec,
        },
    },
};
use rand::{Rng, RngExt, SeedableRng};
use std::marker::PhantomData;

/// Parallel version of [`SinglePoint`](crate::operators::sequential::crossover::SinglePoint).
///
/// Distributes pairs of individuals across pool workers for crossover.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct SinglePoint<T> {
    _marker: PhantomData<T>,
}

impl<T> SinglePoint<T> {
    /// Creates a new parallel `SinglePoint` crossover operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for SinglePoint<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for SinglePoint<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<[T; N], F>: Sync,
{
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[T; N], F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = single_point_crossover(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for SinglePoint<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<Vec<T>, F>: Sync,
{
    fn apply(&self, state: &State<Vec<T>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<T>, F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = single_point_crossover_vec(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

/// Parallel version of [`Uniform`](crate::operators::sequential::crossover::Uniform).
///
/// Distributes pairs of individuals across pool workers for crossover.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct Uniform<T> {
    _marker: PhantomData<T>,
}

impl<T> Uniform<T> {
    /// Creates a new parallel `Uniform` operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Uniform<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for Uniform<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<[T; N], F>: Sync,
{
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[T; N], F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = uniform_crossover(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for Uniform<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<Vec<T>, F>: Sync,
{
    fn apply(&self, state: &State<Vec<T>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<T>, F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = uniform_crossover_vec(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

/// Parallel version of [`TwoPoint`](crate::operators::sequential::crossover::TwoPoint).
///
/// Distributes pairs of individuals across pool workers for crossover.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Clone, Copy)]
pub struct TwoPoint<T> {
    _marker: PhantomData<T>,
}

impl<T> TwoPoint<T> {
    /// Creates a new parallel `TwoPoint` crossover operator.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for TwoPoint<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, F, Fe, R, C, const N: usize> GeneticOperator<[T; N], F, Fe, R, C> for TwoPoint<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<[T; N], F>: Sync,
{
    fn apply(&self, state: &State<[T; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[T; N], F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = two_point_crossover(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

impl<T, F, Fe, R, C> GeneticOperator<Vec<T>, F, Fe, R, C> for TwoPoint<T>
where
    T: Clone + Send + Sync,
    F: Send,
    R: Rng + SeedableRng,
    Fe: Sync,
    C: Sync,
    Individual<Vec<T>, F>: Sync,
{
    fn apply(&self, state: &State<Vec<T>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<T>, F> {
        let individuals = state.population().as_slice();
        let inputs: vecpool::PoolVec<(u64, usize)> = individuals
            .chunks_exact(2)
            .enumerate()
            .map(|(i, _)| (ctx.rng().random::<u64>(), i))
            .collect();

        let results = ctx.pool().map(&inputs, |(seed, idx)| {
            let mut rng = R::seed_from_u64(*seed);
            let base = idx * 2;
            let (c1, c2) = two_point_crossover_vec(
                individuals[base].genome(),
                individuals[base + 1].genome(),
                &mut rng,
            );
            (Individual::new(c1), Individual::new(c2))
        });

        let mut population = Population::with_capacity(inputs.len() * 2);
        for r in results {
            let (c1, c2) = r.expect("pool task panicked");
            population.add(c1);
            population.add(c2);
        }
        Offspring::Multiple(population)
    }
}

/// Parallel version of [`Arithmetic`](crate::operators::sequential::crossover::Arithmetic).
///
/// Distributes pairs of individuals across pool workers for blending.
/// Each task gets its own RNG seeded from the main one.
#[derive(Debug, Default, Clone, Copy)]
pub struct Arithmetic;

impl Arithmetic {
    /// Creates a new parallel `Arithmetic` crossover operator.
    pub fn new() -> Self {
        Self
    }
}

macro_rules! impl_parallel_arithmetic {
    ($float:ty, $cast:expr) => {
        impl<F, Fe, R, C, const N: usize> GeneticOperator<[$float; N], F, Fe, R, C> for Arithmetic
        where
            $float: Send + Sync,
            F: Send,
            R: Rng + SeedableRng,
            Fe: Sync,
            C: Sync,
            Individual<[$float; N], F>: Sync,
        {
            fn apply(&self, state: &State<[$float; N], F>, ctx: &mut Context<Fe, R, C>) -> Offspring<[$float; N], F> {
                let individuals = state.population().as_slice();
                let inputs: vecpool::PoolVec<(u64, usize)> = individuals
                    .chunks_exact(2)
                    .enumerate()
                    .map(|(i, _)| (ctx.rng().random::<u64>(), i))
                    .collect();

                let results = ctx.pool().map(&inputs, |(seed, idx)| {
                    let mut rng = R::seed_from_u64(*seed);
                    let base = idx * 2;
                    let p1 = individuals[base].genome();
                    let p2 = individuals[base + 1].genome();
                    let alpha: f64 = rng.random();

                    let child1: [$float; N] = std::array::from_fn(|i| $cast(alpha * p1[i] as f64 + (1.0 - alpha) * p2[i] as f64));
                    let child2: [$float; N] = std::array::from_fn(|i| $cast((1.0 - alpha) * p1[i] as f64 + alpha * p2[i] as f64));

                    (Individual::new(child1), Individual::new(child2))
                });

                let mut population = Population::with_capacity(inputs.len() * 2);
                for r in results {
                    let (c1, c2) = r.expect("pool task panicked");
                    population.add(c1);
                    population.add(c2);
                }
                Offspring::Multiple(population)
            }
        }

        impl<F, Fe, R, C> GeneticOperator<Vec<$float>, F, Fe, R, C> for Arithmetic
        where
            $float: Send + Sync,
            F: Send,
            R: Rng + SeedableRng,
            Fe: Sync,
            C: Sync,
            Individual<Vec<$float>, F>: Sync,
        {
            fn apply(&self, state: &State<Vec<$float>, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<Vec<$float>, F> {
                let individuals = state.population().as_slice();
                let inputs: vecpool::PoolVec<(u64, usize)> = individuals
                    .chunks_exact(2)
                    .enumerate()
                    .map(|(i, _)| (ctx.rng().random::<u64>(), i))
                    .collect();

                let results = ctx.pool().map(&inputs, |(seed, idx)| {
                    let mut rng = R::seed_from_u64(*seed);
                    let base = idx * 2;
                    let p1 = individuals[base].genome();
                    let p2 = individuals[base + 1].genome();
                    let alpha: f64 = rng.random();
                    let len = p1.len().min(p2.len());

                    let child1: Vec<$float> = (0..len)
                        .map(|i| $cast(alpha * p1[i] as f64 + (1.0 - alpha) * p2[i] as f64))
                        .collect();
                    let child2: Vec<$float> = (0..len)
                        .map(|i| $cast((1.0 - alpha) * p1[i] as f64 + alpha * p2[i] as f64))
                        .collect();

                    (Individual::new(child1), Individual::new(child2))
                });

                let mut population = Population::with_capacity(inputs.len() * 2);
                for r in results {
                    let (c1, c2) = r.expect("pool task panicked");
                    population.add(c1);
                    population.add(c2);
                }
                Offspring::Multiple(population)
            }
        }
    };
}

impl_parallel_arithmetic!(f64, |x: f64| x);
impl_parallel_arithmetic!(f32, |x: f64| x as f32);