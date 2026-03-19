use evolve::{
    algorithm::ga::GeneticAlgorithm,
    fitness::Maximize,
    initialization::Random,
    operators::{combinator::Fill, mutation::RandomReset},
    termination::MaxGenerations,
};
use std::num::NonZero;

#[test]
fn maximize_function() {
    fn func(args: &[u32; 2]) -> usize {
        let x = args[0] as usize;
        let y = args[1] as usize;
        x * x - y
    }

    let rng = rand::rng();
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(100),
        func,
        Fill::from_population_size(RandomReset::new()),
        NonZero::new(500).unwrap(),
        rng,
        Maximize,
    );

    dbg!(ga.run());
}
