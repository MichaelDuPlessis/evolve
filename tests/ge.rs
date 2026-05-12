use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::Maximize,
    ge::{
        fitness::GeFitness,
        grammar::Grammar,
        mapper::map,
        phenotype::{Event, Phenotype, PhenotypeBuilder},
    },
    initialization::RangedRandom,
    operators::sequential::{
        combinator::{Combine, Fill, Pipeline, Weighted},
        crossover::SinglePoint,
        mutation::{deletion::SegmentDeletion, duplication::SegmentDuplication, RandomReset},
        selection::TournamentSelection,
    },
    termination::MaxGenerations,
};
use std::num::NonZero;

struct TerminalCount(usize);

impl Phenotype for TerminalCount {
    type Input = ();
    type Output = usize;
    fn run(&self, _: &()) -> usize { self.0 }
}

#[derive(Default)]
struct CountBuilder(usize);

impl PhenotypeBuilder<&'static str> for CountBuilder {
    type Output = TerminalCount;
    fn push(&mut self, event: Event<&'static str>) {
        if let Event::Terminal(_) = event { self.0 += 1; }
    }
    fn finish(self) -> TerminalCount { TerminalCount(self.0) }
}

fn nz(n: usize) -> NonZero<usize> {
    NonZero::new(n).unwrap()
}

fn nz16(n: u16) -> NonZero<u16> {
    NonZero::new(n).unwrap()
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

#[test]
fn ge_runs_to_completion() {
    let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
        arithmetic_grammar(),
        3,
        |p: &TerminalCount| p.run(&()) as f64,
        -1.0,
    );

    let mut ga = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..20),
        MaxGenerations::new(20),
        fitness,
        Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        ))),
        nz(50),
        rand::rng(),
        Maximize,
    );

    let _result = ga.run();
}

#[test]
fn ge_with_segment_operators() {
    let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
        arithmetic_grammar(),
        3,
        |p: &TerminalCount| p.run(&()) as f64,
        -1.0,
    );

    let mut ga = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..20),
        MaxGenerations::new(10),
        fitness,
        Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::<u8>::new(),
            Weighted::new((
                (RandomReset::<u8>::new(), nz16(3)),
                (SegmentDuplication::<u8>::new(0.2, 50), nz16(1)),
                (SegmentDeletion::<u8>::new(0.2, 3), nz16(1)),
            )),
        ))),
        nz(50),
        rand::rng(),
        Maximize,
    );

    let _result = ga.run();
}

#[test]
fn ge_best_has_valid_phenotype() {
    let grammar = arithmetic_grammar();
    let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
        grammar.clone(),
        3,
        |p: &TerminalCount| p.run(&()) as f64,
        -1.0,
    );

    let mut ga = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..20),
        MaxGenerations::new(30),
        fitness,
        Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        ))),
        nz(50),
        rand::rng(),
        Maximize,
    );

    let result = ga.run();

    let fe = GeFitness::<_, u8, f64, _, CountBuilder>::new(
        grammar.clone(),
        3,
        |p: &TerminalCount| p.run(&()) as f64,
        -1.0,
    );

    let best = result.population.best(&fe, &Maximize);

    let phenotype = map(
        &grammar,
        best.genome(),
        3,
        CountBuilder::default(),
    );
    assert!(
        phenotype.is_some(),
        "Best individual should produce a valid phenotype"
    );
    assert!(
        phenotype.unwrap().run(&()) > 0,
        "Phenotype should have at least one terminal"
    );
}
