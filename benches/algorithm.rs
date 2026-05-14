use criterion::{Criterion, criterion_group, criterion_main};
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    collector::standard::Standard,
    experiment::Experiment,
    fitness::Maximize,
    initialization::Random,
    operators::sequential::{
        combinator::{Combine, Fill, Pipeline},
        crossover::SinglePoint,
        mutation::RandomReset,
        selection::TournamentSelection,
    },
    termination::MaxGenerations,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::num::NonZero;

fn bench_ea_run(c: &mut Criterion) {
    c.bench_function("ea_run_100pop_50gen", |b| {
        b.iter(|| {
            let mut ga = EvolutionaryAlgorithm::new(
                Random::new(),
                MaxGenerations::new(50),
                |g: &[u8; 8]| g.iter().map(|&x| x as u32).sum::<u32>(),
                Fill::from_population_size(Pipeline::new((
                    Combine::new((
                        TournamentSelection::new(NonZero::new(3).unwrap()),
                        TournamentSelection::new(NonZero::new(3).unwrap()),
                    )),
                    SinglePoint::<u8>::new(),
                    RandomReset::<u8>::new(),
                ))),
                NonZero::new(100).unwrap(),
                SmallRng::seed_from_u64(42),
                Maximize,
            );
            ga.run()
        });
    });
}

fn bench_experiment(c: &mut Criterion) {
    c.bench_function("experiment_3_trials", |b| {
        b.iter(|| {
            let mut seed = 42u64;
            Experiment::new(
                move || {
                    seed += 1;
                    EvolutionaryAlgorithm::new(
                        Random::new(),
                        MaxGenerations::new(50),
                        |g: &[u8; 8]| g.iter().map(|&x| x as u32).sum::<u32>(),
                        Fill::from_population_size(Pipeline::new((
                            Combine::new((
                                TournamentSelection::new(NonZero::new(3).unwrap()),
                                TournamentSelection::new(NonZero::new(3).unwrap()),
                            )),
                            SinglePoint::<u8>::new(),
                            RandomReset::<u8>::new(),
                        ))),
                        NonZero::new(100).unwrap(),
                        SmallRng::seed_from_u64(seed),
                        Maximize,
                    )
                },
                3,
                Standard::default,
            )
            .run()
        });
    });
}

criterion_group!(benches, bench_ea_run, bench_experiment);
criterion_main!(benches);
