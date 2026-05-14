use criterion::{Criterion, criterion_group, criterion_main};
use evolve::{
    algorithm::EvolutionaryAlgorithm,
    collector::standard::Standard,
    experiment::Experiment,
    fitness::{GeFitness, Maximize},
    grammar::Grammar,
    initialization::{Random, RangedRandom},
    operators::sequential::{
        combinator::{Combine, Fill, Pipeline, Weighted},
        crossover::SinglePoint,
        mutation::{RandomReset, deletion::SegmentDeletion, duplication::SegmentDuplication},
        selection::TournamentSelection,
    },
    phenotype::{Event, PhenotypeBuilder},
    termination::MaxGenerations,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::num::NonZero;

// --- GE support types ---

struct TerminalCount(usize);

impl TerminalCount {
    fn run(&self, _: &()) -> usize {
        self.0
    }
}

#[derive(Default)]
struct CountBuilder(usize);

impl PhenotypeBuilder<&'static str> for CountBuilder {
    type Output = TerminalCount;
    fn push(&mut self, event: Event<&'static str>) {
        if let Event::Terminal(_) = event {
            self.0 += 1;
        }
    }
    fn finish(self) -> TerminalCount {
        TerminalCount(self.0)
    }
}

fn arithmetic_grammar() -> Grammar<&'static str> {
    Grammar::builder()
        .rule("expr", &[&["expr", "op", "expr"], &["var"], &["const"]])
        .rule("op", &[&["+"], &["-"], &["*"]])
        .rule("var", &[&["x"], &["y"]])
        .rule("const", &[&["1"], &["2"]])
        .start("expr")
        .build()
}

// --- GA benchmarks ---

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

// --- GE benchmarks ---

fn bench_ge_run(c: &mut Criterion) {
    c.bench_function("ge_run_100pop_50gen", |b| {
        b.iter(|| {
            let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
                arithmetic_grammar(),
                3,
                |p: &TerminalCount| p.run(&()) as f64,
                -1.0,
            );
            let mut ga = EvolutionaryAlgorithm::new(
                RangedRandom::<u8>::new(5..20),
                MaxGenerations::new(50),
                fitness,
                Fill::from_population_size(Pipeline::new((
                    Combine::new((
                        TournamentSelection::new(NonZero::new(3).unwrap()),
                        TournamentSelection::new(NonZero::new(3).unwrap()),
                    )),
                    SinglePoint::<u8>::new(),
                    Weighted::new((
                        (RandomReset::<u8>::new(), NonZero::new(3u16).unwrap()),
                        (SegmentDuplication::<u8>::new(0.2, 50), NonZero::new(1u16).unwrap()),
                        (SegmentDeletion::<u8>::new(0.2, 3), NonZero::new(1u16).unwrap()),
                    )),
                ))),
                NonZero::new(100).unwrap(),
                SmallRng::seed_from_u64(42),
                Maximize,
            );
            ga.run()
        });
    });
}

fn bench_ge_experiment(c: &mut Criterion) {
    c.bench_function("ge_experiment_3_trials", |b| {
        b.iter(|| {
            let mut seed = 42u64;
            Experiment::new(
                move || {
                    seed += 1;
                    let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
                        arithmetic_grammar(),
                        3,
                        |p: &TerminalCount| p.run(&()) as f64,
                        -1.0,
                    );
                    EvolutionaryAlgorithm::new(
                        RangedRandom::<u8>::new(5..20),
                        MaxGenerations::new(50),
                        fitness,
                        Fill::from_population_size(Pipeline::new((
                            Combine::new((
                                TournamentSelection::new(NonZero::new(3).unwrap()),
                                TournamentSelection::new(NonZero::new(3).unwrap()),
                            )),
                            SinglePoint::<u8>::new(),
                            Weighted::new((
                                (RandomReset::<u8>::new(), NonZero::new(3u16).unwrap()),
                                (SegmentDuplication::<u8>::new(0.2, 50), NonZero::new(1u16).unwrap()),
                                (SegmentDeletion::<u8>::new(0.2, 3), NonZero::new(1u16).unwrap()),
                            )),
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

criterion_group!(benches, bench_ea_run, bench_experiment, bench_ge_run, bench_ge_experiment);
criterion_main!(benches);
