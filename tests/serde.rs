#![cfg(feature = "serde")]

use evolve::core::{
    individual::Individual, offspring::Offspring, population::Population, state::State,
};

#[test]
fn individual_with_fitness_round_trip() {
    let ind = Individual::<Vec<u8>, u32>::from_parts(vec![1, 2, 3], 6);
    let json = serde_json::to_string(&ind).unwrap();
    let de: Individual<Vec<u8>, u32> = serde_json::from_str(&json).unwrap();
    assert_eq!(de.genome(), &vec![1, 2, 3]);
    assert_eq!(de.try_fitness(), Some(&6));
}

#[test]
fn individual_without_fitness_round_trip() {
    let ind = Individual::<Vec<u8>, u32>::new(vec![4, 5, 6]);
    let json = serde_json::to_string(&ind).unwrap();
    let de: Individual<Vec<u8>, u32> = serde_json::from_str(&json).unwrap();
    assert_eq!(de.genome(), &vec![4, 5, 6]);
    assert_eq!(de.try_fitness(), None);
}

#[test]
fn population_round_trip() {
    let pop = Population::from_individuals(vec![
        Individual::from_parts(1u32, 10i32),
        Individual::from_parts(2, 20),
    ]);
    let json = serde_json::to_string(&pop).unwrap();
    let de: Population<u32, i32> = serde_json::from_str(&json).unwrap();
    assert_eq!(de.len(), 2);
    assert_eq!(de.iter().next().unwrap().try_fitness(), Some(&10));
}

#[test]
fn state_round_trip() {
    let pop = Population::from_individuals(vec![Individual::from_parts(1u32, 10i32)]);
    let state = State::new(pop, 5);
    let json = serde_json::to_string(&state).unwrap();
    let de: State<u32, i32> = serde_json::from_str(&json).unwrap();
    assert_eq!(de.generation(), 5);
    assert_eq!(de.population().len(), 1);
}

#[test]
fn offspring_single_round_trip() {
    let off = Offspring::<u32, i32>::Single(Individual::from_parts(42, 100));
    let json = serde_json::to_string(&off).unwrap();
    let de: Offspring<u32, i32> = serde_json::from_str(&json).unwrap();
    match de {
        Offspring::Single(ind) => {
            assert_eq!(ind.genome(), &42);
            assert_eq!(ind.try_fitness(), Some(&100));
        }
        _ => panic!("expected Single"),
    }
}

#[test]
fn offspring_multiple_round_trip() {
    let pop = Population::from_individuals(vec![
        Individual::from_parts(1u32, 10i32),
        Individual::from_parts(2, 20),
    ]);
    let off = Offspring::Multiple(pop);
    let json = serde_json::to_string(&off).unwrap();
    let de: Offspring<u32, i32> = serde_json::from_str(&json).unwrap();
    match de {
        Offspring::Multiple(p) => assert_eq!(p.len(), 2),
        _ => panic!("expected Multiple"),
    }
}

#[test]
fn basic_run_result_round_trip() {
    use evolve::collector::basic::RunResult;
    let json = r#"{"population":[{"genome":1,"fitness":10}],"generations":5}"#;
    let de: RunResult<u32, i32> = serde_json::from_str(json).unwrap();
    assert_eq!(de.generations(), 5);
    assert_eq!(de.population().len(), 1);
    let re = serde_json::to_string(&de).unwrap();
    assert_eq!(re, json);
}

#[test]
fn standard_run_result_round_trip() {
    use evolve::collector::standard::RunResult;
    let json = r#"{"population":[{"genome":1,"fitness":10}],"generations":3,"total_duration":{"secs":1,"nanos":0},"best_fitness":[10,10,10],"generation_durations":[{"secs":0,"nanos":100},{"secs":0,"nanos":200},{"secs":0,"nanos":300}]}"#;
    let de: RunResult<u32, i32> = serde_json::from_str(json).unwrap();
    assert_eq!(de.generations(), 3);
    assert_eq!(de.best_fitness().len(), 3);
    let re = serde_json::to_string(&de).unwrap();
    assert_eq!(re, json);
}

#[test]
fn generation_record_serializes() {
    use evolve::collector::standard::RunResult;
    let json = r#"{"population":[{"genome":1,"fitness":10}],"generations":1,"total_duration":{"secs":0,"nanos":0},"best_fitness":[10],"generation_durations":[{"secs":0,"nanos":500}]}"#;
    let result: RunResult<u32, i32> = serde_json::from_str(json).unwrap();
    let record = result.generation(0).unwrap();
    let record_json = serde_json::to_string(&record).unwrap();
    assert_eq!(
        record_json,
        r#"{"best_fitness":10,"duration":{"secs":0,"nanos":500}}"#
    );
}
