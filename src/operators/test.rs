use crate::core::{
    context::Context,
    individual::Individual,
    offspring::Offspring,
    population::Population,
    state::State,
};
use crate::fitness::Maximize;
use crate::operators::combinator::{Combine, Fill, Pipeline, Repeat, Weighted};
use crate::operators::crossover::SinglePoint;
use crate::operators::mutation::RandomReset;
use crate::operators::selection::TournamentSelection;
use crate::operators::selection::Elitism;
use crate::operators::GeneticOperator;
use std::num::NonZero;

fn id(g: &[i32; 4]) -> i32 {
    g.iter().sum()
}

fn make_state(genomes: &[[i32; 4]]) -> State<[i32; 4], i32> {
    let pop: Population<[i32; 4], i32> =
        genomes.iter().map(|g| Individual::new(*g)).collect();
    State::new(pop, 0)
}

fn make_ctx(
    rng: &mut impl rand::Rng,
) -> Context<
    '_,
    fn(&[i32; 4]) -> i32,
    impl rand::Rng + '_,
    impl crate::fitness::FitnessComparator<i32> + '_,
> {
    Context::new(&(id as fn(&[i32; 4]) -> i32), rng, &Maximize)
}

// ── TournamentSelection ──

#[test]
fn tournament_selection_returns_single() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let sel = TournamentSelection::new(NonZero::new(2).unwrap());
    assert_eq!(sel.apply(&state, &mut ctx).num_offspring(), 1);
}

// ── RandomReset ──

#[test]
fn random_reset_single_individual_returns_single() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = RandomReset::<i32>::new();
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 1);
    assert!(matches!(offspring, Offspring::Single(_)));
}

#[test]
fn random_reset_multiple_individuals_returns_multiple() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = RandomReset::<i32>::new();
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 2);
    assert!(matches!(offspring, Offspring::Multiple(_)));
}

// ── SinglePoint crossover ──

#[test]
fn single_point_crossover_single_pair() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = SinglePoint::<i32>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 2);
}

#[test]
fn single_point_crossover_even_population() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = SinglePoint::<i32>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

#[test]
fn single_point_crossover_odd_drops_remainder() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = SinglePoint::<i32>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 2);
}

#[test]
fn single_point_crossover_children_contain_parent_genes() {
    let p1 = [0i32, 0, 0, 0];
    let p2 = [1i32, 1, 1, 1];
    let state = make_state(&[p1, p2]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = SinglePoint::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

// ── Repeat ──

#[test]
fn repeat_applies_n_times() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Repeat::new(RandomReset::<i32>::new(), 3);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 6);
}

#[test]
fn repeat_zero_produces_empty() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Repeat::new(RandomReset::<i32>::new(), 0);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 0);
}

// ── Combine ──

#[test]
fn combine_merges_outputs() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Combine::new((RandomReset::<i32>::new(), RandomReset::<i32>::new()));
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

#[test]
fn combine_single_element_tuple() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Combine::new((RandomReset::<i32>::new(),));
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 1);
}

// ── Fill ──

#[test]
fn fill_from_population_size() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Fill::from_population_size(RandomReset::<i32>::new());
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 3);
}

#[test]
fn fill_from_fixed_size() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Fill::from_fixed_size(RandomReset::<i32>::new(), 7);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 7);
}

#[test]
fn fill_does_not_overshoot() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Fill::from_fixed_size(RandomReset::<i32>::new(), 5);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 5);
}

// ── Weighted ──

#[test]
fn weighted_picks_one_operator() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Weighted::new((
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
    ));
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 1);
}

// ── Pipeline ──

#[test]
fn pipeline_single_operator() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Pipeline::new((RandomReset::<i32>::new(),));
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 2);
}

#[test]
fn pipeline_chains_operators() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Pipeline::new((SinglePoint::<i32>::new(), RandomReset::<i32>::new()));
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Elitism ──

#[test]
fn elitism_default_returns_single_best() {
    let state = make_state(&[[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let elite = Elitism::default();
    let offspring = elite.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 1);
    match offspring {
        Offspring::Single(ind) => assert_eq!(*ind.fitness(&id), 3),
        _ => panic!("expected Single"),
    }
}

#[test]
fn elitism_returns_best_n() {
    let state = make_state(&[[1, 0, 0, 0], [5, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let elite = Elitism::new(NonZero::new(3).unwrap());
    let offspring = elite.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 3);
    let pop: Population<_, _> = offspring.into();
    let mut fitnesses: Vec<i32> = pop.iter().map(|i| *i.fitness(&id)).collect();
    fitnesses.sort();
    assert_eq!(fitnesses, vec![3, 4, 5]);
}
