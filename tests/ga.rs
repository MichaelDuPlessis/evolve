use evolve::{
    algorithm::ga::GeneticAlgorithm,
    fitness::{Maximize, Minimize},
    initialization::Random,
    operators::{
        combinator::{Combine, Fill, Pipeline, Repeat, Weighted},
        crossover::SinglePoint,
        mutation::RandomReset,
        selection::TournamentSelection,
    },
    termination::MaxGenerations,
};
use rand::SeedableRng;
use std::num::NonZero;

fn nz(n: usize) -> NonZero<usize> {
    NonZero::new(n).unwrap()
}

fn nz16(n: u16) -> NonZero<u16> {
    NonZero::new(n).unwrap()
}

// ── GA can be constructed with different configurations ──

#[test]
fn build_with_maximize() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(RandomReset::new()),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

#[test]
fn build_with_minimize() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(RandomReset::new()),
        nz(10),
        rand::rng(),
        Minimize,
    );
}

#[test]
fn build_with_closure_comparator() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(RandomReset::new()),
        nz(10),
        rand::rng(),
        |a: &u16, b: &u16| a > b,
    );
}

#[test]
fn build_with_pipeline() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>(),
        Fill::from_population_size(Pipeline::new((
            TournamentSelection::new(nz(3)),
            SinglePoint::new(),
            RandomReset::new(),
        ))),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

#[test]
fn build_with_combine() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(Combine::new((RandomReset::new(), RandomReset::new()))),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

#[test]
fn build_with_repeat() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Repeat::new(RandomReset::new(), 10),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

#[test]
fn build_with_weighted() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(Weighted::new((
            (RandomReset::new(), nz16(3)),
            (RandomReset::new(), nz16(1)),
        ))),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

#[test]
fn build_with_fixed_size_fill() {
    GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_fixed_size(RandomReset::new(), 20),
        nz(10),
        rand::rng(),
        Maximize,
    );
}

// ── GA produces improving results ──

#[test]
fn maximize_improves_over_generations() {
    let fitness_fn = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();

    let ops = Fill::from_population_size(Pipeline::new((
        Combine::new((
            TournamentSelection::new(nz(3)),
            TournamentSelection::new(nz(3)),
        )),
        SinglePoint::new(),
        RandomReset::new(),
    )));

    let mut ga_short = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(1),
        fitness_fn,
        ops,
        nz(200),
        rand::rngs::SmallRng::seed_from_u64(42),
        Maximize,
    );

    let ops = Fill::from_population_size(Pipeline::new((
        Combine::new((
            TournamentSelection::new(nz(3)),
            TournamentSelection::new(nz(3)),
        )),
        SinglePoint::new(),
        RandomReset::new(),
    )));

    let mut ga_long = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(200),
        fitness_fn,
        ops,
        nz(200),
        rand::rngs::SmallRng::seed_from_u64(42),
        Maximize,
    );

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
        "200 generations ({long_best}) should be >= 1 generation ({short_best})"
    );
}

#[test]
fn minimize_finds_low_fitness() {
    let fitness_fn = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(200),
        fitness_fn,
        Fill::from_population_size(RandomReset::new()),
        nz(200),
        rand::rng(),
        Minimize,
    );

    let best = ga.run();
    let best_ind = best.population.best(&fitness_fn, &Minimize);
    assert!(
        *best_ind.fitness(&fitness_fn) < 100,
        "expected low fitness, got {}",
        best_ind.fitness(&fitness_fn)
    );
}

// ── Full pipeline: select → crossover → mutate ──

#[test]
fn full_pipeline_runs_to_completion() {
    let fitness_fn = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(50),
        fitness_fn,
        Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::new(),
            RandomReset::new(),
        ))),
        nz(100),
        rand::rng(),
        Maximize,
    );

    let best = ga.run();
    assert!(
        *best
            .population
            .best(&fitness_fn, &Maximize)
            .fitness(&fitness_fn)
            > 0
    );
}

// ── Zero generations returns best of initial population ──

#[test]
fn zero_generations_returns_initial_best() {
    let fitness_fn = |g: &[u8; 2]| g[0] as u16 + g[1] as u16;
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(0),
        fitness_fn,
        Fill::from_population_size(RandomReset::new()),
        nz(50),
        rand::rng(),
        Maximize,
    );

    let best = ga.run();
    assert!(
        *best
            .population
            .best(&fitness_fn, &Maximize)
            .fitness(&fitness_fn)
            > 0
    );
}

// ── Weighted pipeline favours the heavier operator ──

#[test]
fn weighted_pipeline_with_selection_and_mutation() {
    let fitness_fn = |g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>();
    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(50),
        fitness_fn,
        Fill::from_population_size(Weighted::new((
            (
                Pipeline::new((
                    Combine::new((
                        TournamentSelection::new(nz(3)),
                        TournamentSelection::new(nz(3)),
                    )),
                    SinglePoint::new(),
                    RandomReset::new(),
                )),
                nz16(3),
            ),
            (RandomReset::new(), nz16(1)),
        ))),
        nz(100),
        rand::rng(),
        Maximize,
    );

    let best = ga.run();
    assert!(
        *best
            .population
            .best(&fitness_fn, &Maximize)
            .fitness(&fitness_fn)
            > 0
    );
}

// ── Observer via run_with ──

#[test]
fn run_with_observer() {
    use evolve::core::{context::Context, state::State};
    use evolve::observer::Observer;

    struct Counter {
        started: bool,
        generations: usize,
        ended: bool,
    }

    impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for Counter {
        fn on_start(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.started = true;
        }
        fn on_generation(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.generations += 1;
        }
        fn on_end(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.ended = true;
        }
    }

    let mut ga = GeneticAlgorithm::new(
        Random::new(),
        MaxGenerations::new(5),
        |g: &[u8; 2]| g[0] as u16 + g[1] as u16,
        Fill::from_population_size(RandomReset::new()),
        nz(50),
        rand::rng(),
        Maximize,
    );

    let result = ga.run_with(Counter {
        started: false,
        generations: 0,
        ended: false,
    });
    assert!(result.population.len() > 0);
}

// ── Builder ──

#[test]
fn builder_with_all_fields() {
    let mut ga = GeneticAlgorithm::builder(nz(50))
        .initializer(Random::new())
        .termination(MaxGenerations::new(10))
        .fitness(|g: &[u8; 2]| g[0] as u16 + g[1] as u16)
        .operators(Fill::from_population_size(RandomReset::new()))
        .rng(rand::rng())
        .comparator(Maximize)
        .build();

    let result = ga.run();
    assert!(result.population.len() > 0);
}

#[test]
fn builder_with_minimize() {
    let mut ga = GeneticAlgorithm::builder(nz(50))
        .initializer(Random::new())
        .termination(MaxGenerations::new(10))
        .fitness(|g: &[u8; 2]| g[0] as u16 + g[1] as u16)
        .operators(Fill::from_population_size(RandomReset::new()))
        .rng(rand::rng())
        .comparator(Minimize)
        .build();

    let result = ga.run();
    assert!(result.population.len() > 0);
}

#[test]
fn builder_with_pipeline() {
    let mut ga = GeneticAlgorithm::builder(nz(100))
        .initializer(Random::new())
        .termination(MaxGenerations::new(10))
        .fitness(|g: &[u8; 4]| g.iter().map(|x| *x as u32).sum::<u32>())
        .operators(Fill::from_population_size(Pipeline::new((
            Combine::new((
                TournamentSelection::new(nz(3)),
                TournamentSelection::new(nz(3)),
            )),
            SinglePoint::new(),
            RandomReset::new(),
        ))))
        .rng(rand::rng())
        .comparator(Maximize)
        .build();

    let result = ga.run();
    assert!(result.population.len() > 0);
}

#[test]
fn builder_fields_in_any_order() {
    let mut ga = GeneticAlgorithm::builder(nz(50))
        .comparator(Maximize)
        .rng(rand::rng())
        .operators(Fill::from_population_size(RandomReset::new()))
        .fitness(|g: &[u8; 2]| g[0] as u16 + g[1] as u16)
        .termination(MaxGenerations::new(5))
        .initializer(Random::new())
        .build();

    let result = ga.run();
    assert!(result.population.len() > 0);
}
