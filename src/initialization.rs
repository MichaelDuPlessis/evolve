use crate::{
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluator,
    random::Randomizable,
};
use rand::Rng;
use std::num::NonZero;

/// Trait to initialize the population
pub trait Initializer<G, F, Fe, R, C>
where
    Fe: FitnessEvaluator<G, F>,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F>;
}

/// Create a population by randomly creating their genomes.
#[derive(Debug)]
pub struct Random;

impl Random {
    /// Create a new `Random` intializer.
    pub fn new() -> Self {
        Self
    }
}

impl<G, F, Fe, R, C> Initializer<G, F, Fe, R, C> for Random
where
    Fe: FitnessEvaluator<G, F>,
    G: Randomizable<R>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<G, F> {
        (0..population_size.get())
            .map(|_| Individual::new(G::random(ctx.rng()), ctx.fitness_evaluator()))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::context::Context;
    use crate::fitness::Maximize;

    fn id(g: &[u8; 2]) -> u16 {
        g[0] as u16 + g[1] as u16
    }

    #[test]
    fn random_initializer_creates_correct_size() {
        let mut rng = rand::rng();
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        let pop = Random::new().initialize(NonZero::new(10).unwrap(), &mut ctx);
        assert_eq!(pop.len(), 10);
    }

    #[test]
    fn random_initializer_evaluates_fitness() {
        let mut rng = rand::rng();
        let mut ctx = Context::new(&(id as fn(&[u8; 2]) -> u16), &mut rng, &Maximize);
        let pop = Random::new().initialize(NonZero::new(5).unwrap(), &mut ctx);
        for ind in &pop {
            assert_eq!(*ind.fitness(), ind.genome()[0] as u16 + ind.genome()[1] as u16);
        }
    }
}
