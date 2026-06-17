use criterion::{Criterion, criterion_group, criterion_main};
use evolve::{
    core::{context::Context, individual::Individual, population::Population, state::State},
    fitness::{GeFitness, Maximize},
    grammar::Grammar,
    initialization::{Initializer, Random, RangedRandom},
    operators::{
        GeneticOperator,
        sequential::{
            combinator::{Combine, Fill, Pipeline, Proportional},
            crossover::{Arithmetic, SinglePoint, TwoPoint, Uniform},
            mutation::{
                Creep, Gaussian, Inversion, RandomReset, Scramble, Swap, deletion::SegmentDeletion,
                duplication::SegmentDuplication,
            },
            selection::{Rank, RouletteWheel, Sus, Tournament},
            with_rate::WithRate,
        },
    },
    phenotype::{Event, PhenotypeBuilder},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::num::NonZero;

#[cfg(feature = "parallel")]
use std::sync::LazyLock;
#[cfg(feature = "parallel")]
static RUNTIME: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(4));

/// Create a Context, handling the parallel/non-parallel feature difference.
macro_rules! make_ctx {
    ($fe:expr, $rng:expr, $cmp:expr) => {{
        #[cfg(feature = "parallel")]
        { Context::new($fe, $rng, $cmp, &RUNTIME) }
        #[cfg(not(feature = "parallel"))]
        { Context::new($fe, $rng, $cmp) }
    }};
}

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

#[allow(clippy::ptr_arg)]
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("uniform_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Uniform::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("two_point_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = TwoPoint::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("arithmetic_f64", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        use evolve::random::Randomizable;
        let pop: Population<[f64; 8], f64> = (0..100)
            .map(|_| Individual::new(<[f64; 8]>::random(&mut rng)))
            .collect();
        let state = State::new(pop, 0);
        let fe = |g: &[f64; 8]| g.iter().sum::<f64>();
        let op = Arithmetic::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("gaussian_f64", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        use evolve::random::Randomizable;
        let pop: Population<[f64; 8], f64> = (0..100)
            .map(|_| Individual::new(<[f64; 8]>::random(&mut rng)))
            .collect();
        let state = State::new(pop, 0);
        let fe = |g: &[f64; 8]| g.iter().sum::<f64>();
        let op = Gaussian::new(0.1);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("swap_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Swap::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("inversion_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Inversion::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("creep_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Creep::<u8>::new(5);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("scramble_array", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Scramble::<u8>::new();
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
        let op = Tournament::new(NonZero::new(3).unwrap());
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("roulette_wheel", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop: Population<[u8; 8], f64> = (0..100)
            .map(|_| {
                use evolve::random::Randomizable;
                let g = <[u8; 8]>::random(&mut rng);
                Individual::from_parts(g, g.iter().map(|&x| x as f64).sum())
            })
            .collect();
        let state = State::new(pop, 0);
        let fe = |g: &[u8; 8]| g.iter().map(|&x| x as f64).sum::<f64>();
        let op = RouletteWheel;
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("rank", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Rank;
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("sus_50", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop: Population<[u8; 8], f64> = (0..100)
            .map(|_| {
                use evolve::random::Randomizable;
                let g = <[u8; 8]>::random(&mut rng);
                Individual::from_parts(g, g.iter().map(|&x| x as f64).sum())
            })
            .collect();
        let state = State::new(pop, 0);
        let fe = |g: &[u8; 8]| g.iter().map(|&x| x as f64).sum::<f64>();
        let op = Sus::new(NonZero::new(50).unwrap());
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            Random::new().initialize(NonZero::new(100).unwrap(), &mut ctx)
        });
    });

    group.bench_function("ranged_random_vec_100", |b| {
        let fe = fitness_vec as fn(&Vec<u8>) -> u32;
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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
                Tournament::new(NonZero::new(3).unwrap()),
                Tournament::new(NonZero::new(3).unwrap()),
            )),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        )));
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("pipeline", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = Pipeline::new((
            Tournament::new(NonZero::new(3).unwrap()),
            SinglePoint::<u8>::new(),
            RandomReset::<u8>::new(),
        ));
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("with_rate", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let op = WithRate::new(RandomReset::<u8>::new(), 0.1);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
            op.apply(&state, &mut ctx)
        });
    });

    group.bench_function("proportional", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let pop = make_population_array(&mut rng, 100);
        let state = State::new(pop, 0);
        let fe = fitness_array as fn(&[u8; 8]) -> u32;
        let ops = [
            (RandomReset::<u8>::new(), NonZero::new(2u16).unwrap()),
            (RandomReset::<u8>::new(), NonZero::new(1u16).unwrap()),
        ];
        let op = Proportional::new(&ops[..]);
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut ctx = make_ctx!(&fe, &mut rng, &Maximize);
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

    let ge = GeFitness::<_, u8, f64, _, ExprBuilder, _>::new(grammar, 3, |p: &Expr| p.0, 0.0);

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
