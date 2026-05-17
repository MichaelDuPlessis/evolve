use crate::core::{
    context::Context, individual::Individual, offspring::Offspring, population::Population,
    state::State,
};
use crate::fitness::Maximize;
use crate::operators::GeneticOperator;
use crate::operators::sequential::combinator::{Combine, Fill, Pipeline, Proportional, Repeat, Weighted};
use crate::operators::sequential::crossover::SinglePoint;
use crate::operators::sequential::mutation::RandomReset;
use crate::operators::sequential::identity::Identity;
use crate::operators::sequential::selection::Elitism;
use crate::operators::sequential::selection::Tournament;
use std::num::NonZero;

fn id(g: &[i32; 4]) -> i32 {
    g.iter().sum()
}

fn make_state(genomes: &[[i32; 4]]) -> State<[i32; 4], i32> {
    let pop: Population<[i32; 4], i32> = genomes.iter().map(|g| Individual::new(*g)).collect();
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
    #[cfg(feature = "parallel")]
    {
        use std::sync::LazyLock;
        static RUNTIME: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(id as fn(&[i32; 4]) -> i32), rng, &Maximize, &RUNTIME)
    }
    #[cfg(not(feature = "parallel"))]
    Context::new(&(id as fn(&[i32; 4]) -> i32), rng, &Maximize)
}

// ── Tournament ──

#[test]
fn tournament_selection_returns_single() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let sel = Tournament::new(NonZero::new(2).unwrap());
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
    let state = make_state(&[
        [1, 0, 0, 0],
        [5, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
        [2, 0, 0, 0],
    ]);
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

// ── Combine with Vec ──

#[test]
fn combine_vec_merges_outputs() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Combine::new(vec![
        RandomReset::<i32>::new(),
        RandomReset::<i32>::new(),
        RandomReset::<i32>::new(),
    ]);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 6);
}

// ── Pipeline with Vec ──

#[test]
fn pipeline_vec_chains_operators() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Pipeline::new(vec![RandomReset::<i32>::new(), RandomReset::<i32>::new()]);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Weighted with Vec ──

#[test]
fn weighted_vec_picks_one_operator() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Weighted::new(vec![
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
    ]);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 1);
}

// ── Combine with Box<[O]> ──

#[test]
fn combine_boxed_slice_merges_outputs() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let ops: Box<[RandomReset<i32>]> =
        vec![RandomReset::new(), RandomReset::new()].into_boxed_slice();
    let op = Combine::new(ops);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Pipeline with Box<[O]> ──

#[test]
fn pipeline_boxed_slice_chains_operators() {
    let state = make_state(&[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let ops: Box<[RandomReset<i32>]> =
        vec![RandomReset::new(), RandomReset::new()].into_boxed_slice();
    let op = Pipeline::new(ops);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 4);
}

// ── Weighted with Box<[(O, NonZero<u16>)]> ──

#[test]
fn weighted_boxed_slice_picks_one_operator() {
    let state = make_state(&[[1, 2, 3, 4]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let ops: Box<[(RandomReset<i32>, NonZero<u16>)]> = vec![
        (RandomReset::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::new(), NonZero::new(1u16).unwrap()),
    ]
    .into_boxed_slice();
    let op = Weighted::new(ops);
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 1);
}

// ── SinglePoint crossover for Vec<T> ──

fn id_vec(g: &Vec<i32>) -> i32 {
    g.iter().sum()
}

fn make_state_vec(genomes: &[Vec<i32>]) -> State<Vec<i32>, i32> {
    let pop: Population<Vec<i32>, i32> =
        genomes.iter().map(|g| Individual::new(g.clone())).collect();
    State::new(pop, 0)
}

fn make_ctx_vec(
    rng: &mut impl rand::Rng,
) -> Context<
    '_,
    fn(&Vec<i32>) -> i32,
    impl rand::Rng + '_,
    impl crate::fitness::FitnessComparator<i32> + '_,
> {
    #[cfg(feature = "parallel")]
    {
        use std::sync::LazyLock;
        static RUNTIME: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(id_vec as fn(&Vec<i32>) -> i32), rng, &Maximize, &RUNTIME)
    }
    #[cfg(not(feature = "parallel"))]
    Context::new(&(id_vec as fn(&Vec<i32>) -> i32), rng, &Maximize)
}

#[test]
fn single_point_crossover_vec_single_pair() {
    let state = make_state_vec(&[vec![1, 1, 1, 1], vec![2, 2, 2, 2]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx_vec(&mut rng);
    let op = SinglePoint::<i32>::new();
    assert_eq!(op.apply(&state, &mut ctx).num_offspring(), 2);
}

#[test]
fn single_point_crossover_vec_children_contain_parent_genes() {
    let state = make_state_vec(&[vec![0, 0, 0, 0], vec![1, 1, 1, 1]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx_vec(&mut rng);
    let op = SinglePoint::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

#[test]
fn single_point_crossover_vec_variable_length_parents() {
    let state = make_state_vec(&[vec![1, 2, 3], vec![4, 5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx_vec(&mut rng);
    let op = SinglePoint::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    // Children lengths should sum to parents' total length
    let total: usize = pop.iter().map(|i| i.genome().len()).sum();
    assert_eq!(total, 8); // 3 + 5 = 8
}

// ── SegmentDuplication ──

#[test]
fn segment_duplication_increases_length() {
    use crate::operators::sequential::mutation::duplication::SegmentDuplication;

    let genomes: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4, 5]];
    let pop: Population<Vec<i32>, i32> = genomes.into_iter().map(Individual::new).collect();
    let state = State::new(pop, 0);
    let mut rng = rand::rng();

    fn eval(g: &Vec<i32>) -> i32 {
        g.iter().sum()
    }

    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize);

    let op = SegmentDuplication::<i32>::new(0.5, 10);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.len(), 1);
    assert!(pop.as_slice()[0].genome().len() > 5);
}

#[test]
fn segment_duplication_skips_when_exceeds_max() {
    use crate::operators::sequential::mutation::duplication::SegmentDuplication;

    let genomes: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4, 5]];
    let pop: Population<Vec<i32>, i32> = genomes.into_iter().map(Individual::new).collect();
    let state = State::new(pop, 0);
    let mut rng = rand::rng();

    fn eval(g: &Vec<i32>) -> i32 {
        g.iter().sum()
    }

    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize);

    // max_genome_len = 5, so no duplication can happen
    let op = SegmentDuplication::<i32>::new(0.5, 5);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.as_slice()[0].genome().len(), 5);
}

// ── SegmentDeletion ──

#[test]
fn segment_deletion_decreases_length() {
    use crate::operators::sequential::mutation::deletion::SegmentDeletion;

    let genomes: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]];
    let pop: Population<Vec<i32>, i32> = genomes.into_iter().map(Individual::new).collect();
    let state = State::new(pop, 0);
    let mut rng = rand::rng();

    fn eval(g: &Vec<i32>) -> i32 {
        g.iter().sum()
    }

    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize);

    let op = SegmentDeletion::<i32>::new(0.5, 2);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.len(), 1);
    assert!(pop.as_slice()[0].genome().len() < 8);
}

#[test]
fn segment_deletion_skips_when_at_min_len() {
    use crate::operators::sequential::mutation::deletion::SegmentDeletion;

    let genomes: Vec<Vec<i32>> = vec![vec![1, 2, 3]];
    let pop: Population<Vec<i32>, i32> = genomes.into_iter().map(Individual::new).collect();
    let state = State::new(pop, 0);
    let mut rng = rand::rng();

    fn eval(g: &Vec<i32>) -> i32 {
        g.iter().sum()
    }

    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(eval as fn(&Vec<i32>) -> i32), &mut rng, &Maximize);

    // min_genome_len = 3, genome is already 3, so no deletion
    let op = SegmentDeletion::<i32>::new(0.5, 3);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.as_slice()[0].genome().len(), 3);
}

// ── Identity ──

#[test]
fn identity_returns_population_unchanged() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Identity;
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 2);
    assert!(matches!(offspring, Offspring::Multiple(_)));
}

#[test]
fn identity_transform_returns_population_unchanged() {
    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);
    let op = Identity;
    let offspring = op.transform(state, &mut ctx);
    assert_eq!(offspring.num_offspring(), 3);
    assert!(matches!(offspring, Offspring::Multiple(_)));
}

// ── WithRate ──

#[test]
fn with_rate_zero_passes_all_through() {
    use crate::operators::sequential::with_rate::WithRate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let state = make_state(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx(&mut rng);
    let op = WithRate::new(RandomReset::<i32>::new(), 0.0);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.len(), 3);
    assert_eq!(*pop.as_slice()[0].genome(), [1, 2, 3, 4]);
    assert_eq!(*pop.as_slice()[1].genome(), [5, 6, 7, 8]);
    assert_eq!(*pop.as_slice()[2].genome(), [9, 10, 11, 12]);
}

#[test]
fn with_rate_one_applies_to_all() {
    use crate::operators::sequential::with_rate::WithRate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let state = make_state(&[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx(&mut rng);
    let op = WithRate::new(RandomReset::<i32>::new(), 1.0);
    let offspring = op.apply(&state, &mut ctx);
    let pop = offspring.into_population();
    assert_eq!(pop.len(), 3);
    // With rate 1.0, all individuals should be mutated (not all zeros)
    for ind in &pop {
        assert_ne!(*ind.genome(), [0, 0, 0, 0]);
    }
}

#[test]
#[should_panic]
fn with_rate_panics_on_invalid() {
    use crate::operators::sequential::with_rate::WithRate;
    WithRate::new(RandomReset::<i32>::new(), 1.5);
}

// ── Uniform ──

#[test]
fn uniform_crossover_fixed_produces_valid_offspring() {
    use crate::operators::sequential::crossover::Uniform;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let p1 = [0i32, 0, 0, 0];
    let p2 = [1i32, 1, 1, 1];
    let state = make_state(&[p1, p2]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx(&mut rng);
    let op = Uniform::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

#[test]
fn uniform_crossover_vec_produces_valid_offspring() {
    use crate::operators::sequential::crossover::Uniform;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let state = make_state_vec(&[vec![0, 0, 0, 0], vec![1, 1, 1, 1]]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx_vec(&mut rng);
    let op = Uniform::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

// ── Gaussian ──

#[test]
fn gaussian_mutation_modifies_genome() {
    use crate::operators::sequential::mutation::Gaussian;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let pop: Population<[f64; 4], f64> = vec![[0.0; 4], [0.0; 4], [0.0; 4]]
        .into_iter()
        .map(Individual::new)
        .collect();
    let state = State::new(pop, 0);

    fn fe(g: &[f64; 4]) -> f64 {
        g.iter().sum()
    }

    let mut rng = SmallRng::seed_from_u64(42);
    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize);

    let op = Gaussian::new(1.0);
    let offspring = op.apply(&state, &mut ctx);
    let result = offspring.into_population();
    assert!(result.iter().any(|ind| ind.genome().iter().any(|&g| g != 0.0)));
}

#[test]
fn gaussian_mutation_transform_modifies_genome() {
    use crate::operators::sequential::mutation::Gaussian;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let pop: Population<[f64; 4], f64> = vec![[0.0; 4], [0.0; 4], [0.0; 4]]
        .into_iter()
        .map(Individual::new)
        .collect();
    let state = State::new(pop, 0);

    fn fe(g: &[f64; 4]) -> f64 {
        g.iter().sum()
    }

    let mut rng = SmallRng::seed_from_u64(42);
    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize);

    let op = Gaussian::new(1.0);
    let offspring = op.transform(state, &mut ctx);
    let result = offspring.into_population();
    assert!(result.iter().any(|ind| ind.genome().iter().any(|&g| g != 0.0)));
}

// ── RouletteWheel ──

#[test]
fn roulette_wheel_selects_proportionally() {
    use crate::operators::sequential::selection::RouletteWheel;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    // Individual with fitness 99 vs fitness 1 — should be selected ~99% of the time
    let pop: Population<[i32; 4], i32> = vec![
        Individual::from_parts([1, 0, 0, 0], 1),
        Individual::from_parts([9, 9, 9, 9], 99),
    ]
    .into_iter()
    .collect();
    let state = State::new(pop, 0);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx(&mut rng);

    let op = RouletteWheel;
    let mut high_count = 0;
    for _ in 0..100 {
        let offspring = op.apply(&state, &mut ctx);
        if let Offspring::Single(ind) = offspring {
            if *ind.genome() == [9, 9, 9, 9] {
                high_count += 1;
            }
        }
    }
    assert!(
        high_count > 80,
        "high fitness individual selected {high_count}/100 times"
    );
}

// ── Proportional ──

#[test]
fn proportional_output_matches_input_size() {
    let state = make_state(&[[1, 2, 3, 4]; 12]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let ops = [
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(2u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(3u16).unwrap()),
    ];
    let op = Proportional::new(ops.as_slice());
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 12);
}

#[test]
fn proportional_handles_remainder() {
    // 7 individuals with weights 1:1 => 3 + 4 (last gets remainder)
    let state = make_state(&[[1, 2, 3, 4]; 7]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let ops = [
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
    ];
    let op = Proportional::new(ops.as_slice());
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 7);
}

#[test]
fn proportional_single_operator() {
    let state = make_state(&[[1, 2, 3, 4]; 5]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let ops = [(
        RandomReset::<i32>::new(),
        NonZero::new(3u16).unwrap(),
    )];
    let op = Proportional::new(ops.as_slice());
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 5);
}

#[test]
fn proportional_slice_reference() {
    let state = make_state(&[[1, 2, 3, 4]; 6]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let ops = vec![
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(2u16).unwrap()),
    ];
    let op = Proportional::new(&ops[..]);
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 6);
}

#[test]
fn proportional_fixed_size() {
    let state = make_state(&[[1, 2, 3, 4]; 4]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let ops = [
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(2u16).unwrap()),
    ];
    let op = Proportional::with_size(ops.as_slice(), 9);
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 9);
}

// ── Proportional with tuples ──

#[test]
fn proportional_tuple_two_operators() {
    let state = make_state(&[[1, 2, 3, 4]; 12]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let op = Proportional::new((
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(2u16).unwrap()),
    ));
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 12);
}

#[test]
fn proportional_tuple_with_size() {
    let state = make_state(&[[1, 2, 3, 4]; 6]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let op = Proportional::with_size((
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(3u16).unwrap()),
    ), 20);
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 20);
}

#[test]
fn proportional_tuple_three_operators() {
    let state = make_state(&[[1, 2, 3, 4]; 9]);
    let mut rng = rand::rng();
    let mut ctx = make_ctx(&mut rng);

    let op = Proportional::new((
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
        (RandomReset::<i32>::new(), NonZero::new(1u16).unwrap()),
    ));
    let offspring = op.apply(&state, &mut ctx);
    assert_eq!(offspring.into_population().len(), 9);
}

// ── TwoPoint ──

#[test]
fn two_point_crossover_fixed_produces_valid_offspring() {
    use crate::operators::sequential::crossover::TwoPoint;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let p1 = [0i32, 0, 0, 0];
    let p2 = [1i32, 1, 1, 1];
    let state = make_state(&[p1, p2]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx(&mut rng);
    let op = TwoPoint::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

#[test]
fn two_point_crossover_vec_produces_valid_offspring() {
    use crate::operators::sequential::crossover::TwoPoint;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let state = make_state_vec(&[vec![0, 0, 0, 0], vec![1, 1, 1, 1]]);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ctx = make_ctx_vec(&mut rng);
    let op = TwoPoint::<i32>::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for gene in ind.genome() {
            assert!(*gene == 0 || *gene == 1);
        }
    }
}

// ── Arithmetic ──

#[test]
fn arithmetic_crossover_f64_blends_parents() {
    use crate::operators::sequential::crossover::Arithmetic;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let pop: Population<[f64; 4], f64> = vec![
        Individual::new([0.0, 0.0, 0.0, 0.0]),
        Individual::new([1.0, 1.0, 1.0, 1.0]),
    ]
    .into_iter()
    .collect();
    let state = State::new(pop, 0);

    fn fe(g: &[f64; 4]) -> f64 {
        g.iter().sum()
    }

    let mut rng = SmallRng::seed_from_u64(42);
    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(fe as fn(&[f64; 4]) -> f64), &mut rng, &Maximize);

    let op = Arithmetic::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for &gene in ind.genome() {
            assert!(gene >= 0.0 && gene <= 1.0, "gene {gene} not in [0, 1]");
        }
    }
}

#[test]
fn arithmetic_crossover_f32_produces_valid_offspring() {
    use crate::operators::sequential::crossover::Arithmetic;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let pop: Population<[f32; 4], f32> = vec![
        Individual::new([0.0f32, 2.0, 4.0, 6.0]),
        Individual::new([1.0f32, 3.0, 5.0, 7.0]),
    ]
    .into_iter()
    .collect();
    let state = State::new(pop, 0);

    fn fe(g: &[f32; 4]) -> f32 {
        g.iter().sum()
    }

    let mut rng = SmallRng::seed_from_u64(99);
    #[cfg(feature = "parallel")]
    let mut ctx = {
        use std::sync::LazyLock;
        static RT: LazyLock<pooled::Runtime> = LazyLock::new(|| pooled::Runtime::new(1));
        Context::new(&(fe as fn(&[f32; 4]) -> f32), &mut rng, &Maximize, &RT)
    };
    #[cfg(not(feature = "parallel"))]
    let mut ctx = Context::new(&(fe as fn(&[f32; 4]) -> f32), &mut rng, &Maximize);

    let op = Arithmetic::new();
    let pop = op.apply(&state, &mut ctx).into_population();
    assert_eq!(pop.len(), 2);
    for ind in &pop {
        for (i, &gene) in ind.genome().iter().enumerate() {
            let lo = [0.0f32, 2.0, 4.0, 6.0][i];
            let hi = [1.0f32, 3.0, 5.0, 7.0][i];
            assert!(gene >= lo && gene <= hi, "gene {gene} not in [{lo}, {hi}]");
        }
    }
}