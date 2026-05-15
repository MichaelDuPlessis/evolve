use criterion::{Criterion, criterion_group, criterion_main};
use evolve::{
    core::{context::Context, individual::Individual, population::Population, state::State},
    fitness::{GeFitness, Maximize},
    grammar::Grammar,
    initialization::{Initializer, Random, RangedRandom},
    operators::{
        GeneticOperator,
        sequential::{
            combinator::{Combine, Fill, Pipeline},
            crossover::SinglePoint,
            mutation::{deletion::SegmentDeletion, duplication::SegmentDuplication, RandomReset},
            selection::TournamentSelection,
        },
    },
    phenotype::{Event, PhenotypeBuilder},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::num::NonZero;

fn make_population_array(rng: &mut SmallRng, size: usize) -> Population<[u8; 8], u32> {
    use evolve::random::Randomizable;
    (0..size)
        .map(|_| Individual::new(<[u8; 8]>::random(rng)))
        .collect()
}

fn make_population_vec(rng: &mut SmallRng, size: usize) -> Population<Vec<u8>, u32> {
    use evolve::random::Randomizable;
    use rand::RngExt;
    (0..size)
        .map(|_| {
            let len = rng.random_range(50..200);
            Individual::new((0..len).map(|_| u8::random(rng)).collect())
        })
        .collect()
}

fn fitness_array(g: &[u8; 8]) -> u32 {
    g.iter().map(|&x| x as u32).sum()
}

fn fitness_vec(g: &Vec<u8>) -> u32 {
    g.iter().map(|&x| x as u32).sum()
}

fn bench_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");

    group.bench_function("single_point_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = SinglePoint::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("single_point_vec", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_vec(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_vec as fn(&Vec<u8>) -> u32;
        let op = SinglePoint::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.finish();
}

fn bench_mutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation");

    group.bench_function("random_reset_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = RandomReset::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("segment_duplication_vec", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_vec(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_vec as fn(&Vec<u8>) -> u32;
        let op = SegmentDuplication::<u8>::new(0.25, 400);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("segment_deletion_vec", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_vec(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_vec as fn(&Vec<u8>) -> u32;
        let op = SegmentDeletion::<u8>::new(0.25, 10);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.finish();
}

fn bench_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection");

    group.bench_function("tournament_size_3", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = TournamentSelection::new(NonZero::new(3).unwrap());
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.finish();
}

fn bench_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("initialization");

    group.bench_function("random_array_100", |b| {
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            Random::new().initialize(NonZero::new(100).unwrap(), &mut ctx)
        });
    });

    group.bench_function("ranged_random_vec_100", |b| {
        let fe = fitness_vec as fn(&Vec<u8>) -> u32;
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            RangedRandom::<u8>::new(50..200).initialize(NonZero::new(100).unwrap(), &mut ctx)
        });
    });

    group.finish();
}

fn bench_combinators(c: &mut Criterion) {
    let mut group = c.benchmark_group("combinators");

    group.bench_function("fill", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(NonZero::new(3).unwrap()),
                TournamentSelection::new(NonZero::new(3).unwrap()),
            )),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        )));
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("pipeline", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Pipeline::new((
            TournamentSelection::new(NonZero::new(3).unwrap()),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        ));
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.finish();
}

// GE mapping benchmark
#[derive(Debug)]
struct Expr(f64);

#[derive(Default)]
struct ExprBuilder(Vec<&'static str>);

impl PhenotypeBuilder<&'static str> for ExprBuilder {
    type Output = Expr;
    fn push(&mut self, event: Event<&'static str>) {
        if let Event::Terminal(t) = event {
            self.0.push(t);
        }
    }
    fn finish(self) -> Expr {
        // Simple evaluation: count terminals as a proxy
        Expr(self.0.len() as f64)
    }
}

fn bench_ge_mapping(c: &mut Criterion) {
    let grammar = Grammar::builder()
        .rule("expr", &[&["expr", "op", "expr"], &["num"]])
        .rule("op", &[&["+"], &["-"], &["*"]])
        .rule("num", &[&["1"], &["2"], &["3"]])
        .start("expr")
        .build();

    let ge = GeFitness::<_, u8, f64, _, ExprBuilder>::new(
        grammar,
        3,
        |p: &Expr| p.0,
        0.0,
    );

    c.bench_function("ge_mapping", |b| {
        let genome: Vec<u8> = vec![0, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0];
        b.iter(|| ge.phenotype(&genome));
    });
}

#[cfg(feature = "parallel")]
fn bench_parallel(c: &mut Criterion) {
    use evolve::operators::parallel::{
        combinator::Fill as ParFill, mutation::RandomReset as ParRandomReset,
    };

    let mut group = c.benchmark_group("parallel");

    group.bench_function("fill", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let runtime = pooled::Runtime::new(4);
        let op = ParFill::new(RandomReset::<u8>::new(), 100);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("random_reset", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let runtime = pooled::Runtime::new(4);
        let op = ParRandomReset::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);
            op.apply(&state, &mut ctx)
        });
    });

    group.finish();
}

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_crossover,
    bench_mutation,
    bench_selection,
    bench_initialization,
    bench_combinators,
    bench_ge_mapping,
);

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_crossover,
    bench_mutation,
    bench_selection,
    bench_initialization,
    bench_combinators,
    bench_ge_mapping,
    bench_parallel,
);

criterion_main!(benches);
