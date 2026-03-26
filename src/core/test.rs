use crate::core::{
    individual::Individual, offspring::Offspring, population::Population, state::State,
};
use crate::fitness::{FitnessComparator, Maximize, Minimize};

fn id(g: &i32) -> i32 {
    *g
}

// ── Individual ──

#[test]
fn individual_stores_genome_and_fitness() {
    let ind = Individual::new(10);
    assert_eq!(*ind.genome(), 10);
    assert_eq!(*ind.fitness(&id), 10);
}

#[test]
fn individual_with_closure_evaluator() {
    let ind = Individual::new(3);
    assert_eq!(*ind.fitness(&|g: &i32| g * g), 9);
}

#[test]
fn try_fitness_returns_none_before_evaluation() {
    let ind = Individual::<i32, i32>::new(10);
    assert!(ind.try_fitness().is_none());
}

#[test]
fn try_fitness_returns_some_after_evaluation() {
    let ind = Individual::new(10);
    ind.fitness(&id);
    assert_eq!(ind.try_fitness(), Some(&10));
}

#[test]
fn from_parts_has_fitness_immediately() {
    let ind = Individual::from_parts(10, 42);
    assert_eq!(ind.try_fitness(), Some(&42));
}

#[test]
fn fitness_is_cached() {
    use std::cell::Cell;
    let call_count = Cell::new(0u32);
    let counting_fe = |g: &i32| {
        call_count.set(call_count.get() + 1);
        *g
    };
    let ind = Individual::new(10);
    ind.fitness(&counting_fe);
    ind.fitness(&counting_fe);
    assert_eq!(call_count.get(), 1);
}

// ── Population ──

#[test]
fn population_new_is_empty() {
    let pop = Population::<i32, i32>::new();
    assert!(pop.is_empty());
    assert_eq!(pop.len(), 0);
}

#[test]
fn population_add_and_len() {
    let mut pop: Population<i32, i32> = Population::new();
    pop.add(Individual::new(1));
    pop.add(Individual::new(2));
    assert_eq!(pop.len(), 2);
}

#[test]
fn population_best_maximize() {
    let mut pop = Population::new();
    for i in [1, 5, 3] {
        pop.add(Individual::new(i));
    }
    assert_eq!(*pop.best(&id, &Maximize).fitness(&id), 5);
}

#[test]
fn population_best_minimize() {
    let mut pop = Population::new();
    for i in [1, 5, 3] {
        pop.add(Individual::new(i));
    }
    assert_eq!(*pop.best(&id, &Minimize).fitness(&id), 1);
}

#[test]
fn population_merge() {
    let mut pop1: Population<i32, i32> = Population::new();
    pop1.add(Individual::new(1));
    let mut pop2: Population<i32, i32> = Population::new();
    pop2.add(Individual::new(2));
    pop2.add(Individual::new(3));
    pop1.merge(pop2);
    assert_eq!(pop1.len(), 3);
}

#[test]
fn population_cull() {
    let mut pop: Population<i32, i32> = (0..10).map(|i| Individual::new(i)).collect();
    pop.cull(3);
    assert_eq!(pop.len(), 3);
}

#[test]
fn population_from_iterator() {
    let pop: Population<i32, i32> = (0..5).map(|i| Individual::new(i)).collect();
    assert_eq!(pop.len(), 5);
}

#[test]
fn population_into_iter() {
    let mut pop: Population<i32, i32> = Population::new();
    pop.add(Individual::new(1));
    pop.add(Individual::new(2));
    let genomes: Vec<i32> = pop.into_iter().map(|i| *i.genome()).collect();
    assert_eq!(genomes, vec![1, 2]);
}

#[test]
fn population_add_offspring_single() {
    let mut pop: Population<i32, i32> = Population::new();
    pop.add_offspring(Offspring::Single(Individual::new(42)));
    assert_eq!(pop.len(), 1);
    assert_eq!(*pop.as_slice()[0].fitness(&id), 42);
}

#[test]
fn population_add_offspring_multiple() {
    let mut pop: Population<i32, i32> = Population::new();
    let mut inner: Population<i32, i32> = Population::new();
    inner.add(Individual::new(1));
    inner.add(Individual::new(2));
    pop.add_offspring(Offspring::Multiple(inner));
    assert_eq!(pop.len(), 2);
}

// ── Offspring ──

#[test]
fn offspring_single_num_offspring() {
    let o = Offspring::Single(Individual::<i32, i32>::new(7));
    assert_eq!(o.num_offspring(), 1);
}

#[test]
fn offspring_multiple_num_offspring() {
    let mut pop = Population::new();
    pop.add(Individual::<i32, i32>::new(1));
    pop.add(Individual::new(2));
    assert_eq!(Offspring::Multiple(pop).num_offspring(), 2);
}

#[test]
fn offspring_into_population() {
    let o = Offspring::Single(Individual::<i32, i32>::new(10));
    let pop = o.into_population();
    assert_eq!(pop.len(), 1);
    assert_eq!(*pop.as_slice()[0].genome(), 10);
}

// ── State ──

#[test]
fn state_generation_and_population() {
    let mut pop = Population::new();
    pop.add(Individual::<i32, i32>::new(1));
    let state = State::new(pop, 5);
    assert_eq!(state.generation(), 5);
    assert_eq!(state.population().len(), 1);
}

#[test]
fn state_with_population_preserves_generation() {
    let state = State::<i32, i32>::new(Population::new(), 10);
    let mut pop = Population::new();
    pop.add(Individual::new(1));
    let new_state = state.with_population(pop);
    assert_eq!(new_state.generation(), 10);
    assert_eq!(new_state.population().len(), 1);
}

// ── Fitness ──

#[test]
fn maximize_prefers_higher() {
    assert!(Maximize.is_better(&10, &5));
    assert!(!Maximize.is_better(&5, &10));
    assert!(!Maximize.is_better(&5, &5));
}

#[test]
fn minimize_prefers_lower() {
    assert!(Minimize.is_better(&5, &10));
    assert!(!Minimize.is_better(&10, &5));
    assert!(!Minimize.is_better(&5, &5));
}

#[test]
fn closure_as_fitness_comparator() {
    let cmp = |a: &i32, b: &i32| a > b;
    assert!(cmp.is_better(&10, &5));
    assert!(!cmp.is_better(&5, &10));
}
