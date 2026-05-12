#![cfg(feature = "parallel")]

use evolve::{
    algorithm::EvolutionaryAlgorithm,
    core::{context::Context, individual::Individual, population::Population, state::State},
    fitness::Maximize,
    initialization::Random,
    operators::{
        GeneticOperator,
        parallel::{
            combinator::{Fill, Repeat},
            crossover::SinglePoint,
            mutation::RandomReset,
        },
    },
    termination::MaxGenerations,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::num::NonZero;

fn make_state(genomes: &[[u8; 4]]) -> State<[u8; 4], u32> {
    let pop: Population<[u8; 4], u32> = genomes.iter().map(|g| Individual::new(*g)).collect();
    State::new(pop, 0)
}

fn nz(n: usize) -> NonZero<usize> {
    NonZero::new(n).unwrap()
}

// ── Parallel mutation ──

#[test]
fn parallel_random_reset_preserves_population_size() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let op = RandomReset::<u8>::new();
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 3);
}

// ── Parallel crossover ──

#[test]
fn parallel_single_point_even_population() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]);
    let op = SinglePoint::<u8>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Parallel Fill ──

#[test]
fn parallel_fill_produces_exact_target_size() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let op = Fill::new(RandomReset::<u8>::new(), 20);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 20);
}

// ── Parallel Repeat ──

#[test]
fn parallel_repeat_produces_output() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let op = Repeat::new(RandomReset::<u8>::new(), 5);
    assert!(op.apply(&state, &mut ctx).num_offspring() >= 5);
}

// ── Full GA with parallel operators ──

#[test]
fn parallel_ga_improves_over_generations() {
    let fitness_fn = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();

    let mut ga_short = EvolutionaryAlgorithm::builder(nz(100))
        .initializer(Random::new())
        .termination(MaxGenerations::new(1))
        .fitness(fitness_fn)
        .operators(Fill::new(RandomReset::<u8>::new(), 100))
        .rng(SmallRng::seed_from_u64(42))
        .comparator(Maximize)
        .runtime(pooled::Runtime::new(2))
        .build();

    let mut ga_long = EvolutionaryAlgorithm::builder(nz(100))
        .initializer(Random::new())
        .termination(MaxGenerations::new(100))
        .fitness(fitness_fn)
        .operators(Fill::new(RandomReset::<u8>::new(), 100))
        .rng(SmallRng::seed_from_u64(42))
        .comparator(Maximize)
        .runtime(pooled::Runtime::new(2))
        .build();

    let short_best = *ga_short
        .run()
        .population
        .best(&fitness_fn, &Maximize)
        .fitness(&fitness_fn);
    let long_best = *ga_long
        .run()
        .population
        .best(&fitness_fn, &Maximize)
        .fitness(&fitness_fn);

    assert!(
        long_best >= short_best,
        "100 generations ({long_best}) should be >= 1 generation ({short_best})"
    );
}

// ── Parallel Combine ──

#[test]
fn parallel_combine_runs_all_operators() {
    use evolve::operators::parallel::combinator::Combine;

    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);

    // 3 operators, each produces 3 mutated individuals = 9 total
    let op = Combine::new(vec![
        RandomReset::<u8>::new(),
        RandomReset::<u8>::new(),
        RandomReset::<u8>::new(),
    ]);
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 9);
}

// ── Mutation actually mutates ──

#[test]
fn parallel_mutation_changes_genomes() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(99);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let genomes = [[0u8; 4]; 10];
    let state = make_state(&genomes);
    let op = RandomReset::<u8>::new();
    let pop = op.apply(&state, &mut ctx).into_population();

    // At least one individual should differ from all-zeros
    let any_changed = pop.iter().any(|ind| *ind.genome() != [0u8; 4]);
    assert!(any_changed, "mutation should change at least one genome");
}

// ── Crossover actually recombines ──

#[test]
fn parallel_crossover_produces_recombined_children() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(99);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[0, 0, 0, 0], [255, 255, 255, 255]]);
    let op = SinglePoint::<u8>::new();
    let pop = op.apply(&state, &mut ctx).into_population();

    // Children should not both be identical to parents (crossover happened)
    let child1 = pop.as_slice()[0].genome();
    let child2 = pop.as_slice()[1].genome();
    let is_recombined = *child1 != [0, 0, 0, 0] || *child2 != [255, 255, 255, 255];
    assert!(is_recombined, "crossover should recombine parent genomes");
}

// ── Parallel Combine with Box<[O]> ──

#[test]
fn parallel_combine_boxed_slice() {
    use evolve::operators::parallel::combinator::Combine;

    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let ops: Box<[RandomReset<u8>]> =
        vec![RandomReset::new(), RandomReset::new()].into_boxed_slice();
    let op = Combine::new(ops);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Deterministic results with same seed ──

#[test]
fn parallel_mutation_deterministic_with_same_seed() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();

    let state = make_state(&[[10, 20, 30, 40], [50, 60, 70, 80]]);
    let op = RandomReset::<u8>::new();

    let mut rng1 = SmallRng::seed_from_u64(123);
    let mut ctx1 = Context::new(&fe, &mut rng1, &Maximize, &runtime);
    let result1 = op.apply(&state, &mut ctx1).into_population();

    let mut rng2 = SmallRng::seed_from_u64(123);
    let mut ctx2 = Context::new(&fe, &mut rng2, &Maximize, &runtime);
    let result2 = op.apply(&state, &mut ctx2).into_population();

    for (a, b) in result1.iter().zip(result2.iter()) {
        assert_eq!(a.genome(), b.genome());
    }
}

// ── Larger population exercises chunking ──

#[test]
fn parallel_mutation_large_population() {
    let runtime = pooled::Runtime::new(4);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let genomes: Vec<[u8; 4]> = (0..200).map(|i| [i as u8; 4]).collect();
    let pop: Population<[u8; 4], u32> = genomes.iter().map(|g| Individual::new(*g)).collect();
    let state = State::new(pop, 0);

    let op = RandomReset::<u8>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 200);
}

#[test]
fn parallel_crossover_large_population() {
    let runtime = pooled::Runtime::new(4);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let genomes: Vec<[u8; 4]> = (0..100).map(|i| [i as u8; 4]).collect();
    let pop: Population<[u8; 4], u32> = genomes.iter().map(|g| Individual::new(*g)).collect();
    let state = State::new(pop, 0);

    let op = SinglePoint::<u8>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 100);
}

// ── Crossover with odd-sized population ──

#[test]
fn parallel_crossover_odd_population_drops_remainder() {
    let runtime = pooled::Runtime::new(2);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
    let op = SinglePoint::<u8>::new();
    // 3 individuals = 1 pair + 1 remainder, so 2 children
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 2);
}

// ── Fill with larger target ──

#[test]
fn parallel_fill_large_target() {
    let runtime = pooled::Runtime::new(4);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let op = Fill::new(RandomReset::<u8>::new(), 500);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 500);
}

// ── Repeat with larger count ──

#[test]
fn parallel_repeat_large_count() {
    let runtime = pooled::Runtime::new(4);
    let fe = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = Context::new(&fe, &mut rng, &Maximize, &runtime);

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let op = Repeat::new(RandomReset::<u8>::new(), 50);
    // Each rep produces 2 individuals (one per input), so 50 * 2 = 100
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 100);
}
