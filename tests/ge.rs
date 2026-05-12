use evolve::{
    algorithm::EvolutionaryAlgorithm,
    fitness::{FitnessEvaluator, GeFitness, Maximize},
    grammar::Grammar,
    initialization::RangedRandom,
    operators::sequential::{
        combinator::{Combine, Fill, Pipeline, Weighted},
        crossover::SinglePoint,
        mutation::{RandomReset, deletion::SegmentDeletion, duplication::SegmentDuplication},
        selection::TournamentSelection,
    },
    phenotype::bytecode::{Bytecode, BytecodeBuilder, Instruction},
    phenotype::{Event, PhenotypeBuilder},
    termination::MaxGenerations,
};
use std::num::NonZero;

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

    let best = result.population().best(&fe, &Maximize);
    let fitness = fe.evaluate(best.genome());
    assert!(
        fitness > 0.0,
        "Best individual should produce a valid phenotype with at least one terminal"
    );
}

// --- Bytecode integration test ---

#[derive(Clone, PartialEq, Eq, Hash)]
enum Op {
    Push1,
    Push2,
    Add,
    // Used as non-terminal names
    Prog,
    Instr,
}

impl Instruction for Op {
    type Value = i32;
    type Input = ();
    fn execute(&self, stack: &mut Vec<i32>, _: &()) {
        match self {
            Op::Push1 => stack.push(1),
            Op::Push2 => stack.push(2),
            Op::Add => {
                let b = stack.pop().unwrap_or(0);
                let a = stack.pop().unwrap_or(0);
                stack.push(a + b);
            }
            _ => {}
        }
    }
}

fn op_grammar() -> Grammar<Op> {
    Grammar::builder()
        .rule(Op::Prog, &[&[Op::Instr, Op::Prog], &[Op::Instr]])
        .rule(Op::Instr, &[&[Op::Push1], &[Op::Push2], &[Op::Add]])
        .start(Op::Prog)
        .build()
}

#[test]
fn ge_bytecode_integration() {
    let grammar = op_grammar();

    let fitness = GeFitness::<_, u8, i32, _, BytecodeBuilder<Op>>::new(
        grammar.clone(),
        2,
        |p: &Bytecode<Op>| p.run(&()),
        i32::MIN,
    );

    let mut ga = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..15),
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

    let result = ga.run();

    let fe = GeFitness::<_, u8, i32, _, BytecodeBuilder<Op>>::new(
        grammar.clone(),
        2,
        |p: &Bytecode<Op>| p.run(&()),
        i32::MIN,
    );

    let best = result.population().best(&fe, &Maximize);
    let fitness = fe.evaluate(best.genome());
    assert!(
        fitness > i32::MIN,
        "Best individual should produce a valid Bytecode"
    );
}

#[test]
fn ge_improves_fitness_over_generations() {
    use rand::SeedableRng;

    let fitness_fn = |p: &TerminalCount| p.run(&()) as f64;

    let make_fitness =
        || GeFitness::<_, u8, f64, _, CountBuilder>::new(arithmetic_grammar(), 3, fitness_fn, -1.0);

    let ops = || {
        Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        )))
    };

    let mut ga_short = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..20),
        MaxGenerations::new(1),
        make_fitness(),
        ops(),
        nz(100),
        rand::rngs::SmallRng::seed_from_u64(42),
        Maximize,
    );

    let short_result = ga_short.run();

    let mut ga_long = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(5..20),
        MaxGenerations::new(100),
        make_fitness(),
        ops(),
        nz(100),
        rand::rngs::SmallRng::seed_from_u64(42),
        Maximize,
    );

    let long_result = ga_long.run();

    let fe = make_fitness();
    let short_best = fe.evaluate(short_result.population().best(&fe, &Maximize).genome());
    let long_best = fe.evaluate(long_result.population().best(&fe, &Maximize).genome());

    eprintln!("Short best: {short_best}, Long best: {long_best}");

    assert!(
        long_best >= short_best,
        "100 generations ({long_best}) should be >= 1 generation ({short_best})"
    );
}

#[test]
fn ge_with_fixed_length_genome() {
    let fitness = GeFitness::<_, u8, f64, _, CountBuilder>::new(
        arithmetic_grammar(),
        3,
        |p: &TerminalCount| p.run(&()) as f64,
        -1.0,
    );

    let mut ea = EvolutionaryAlgorithm::new(
        RangedRandom::<u8>::new(20..21),
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

    let _result = ea.run();
}
