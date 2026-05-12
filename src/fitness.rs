//! Fitness evaluation and comparison.
//!
//! Defines [`FitnessEvaluator`] for computing fitness and [`FitnessComparator`]
//! for determining which fitness value is better. Both traits have blanket
//! implementations for closures, and the module provides [`Maximize`] and
//! [`Minimize`] comparators out of the box.

/// Evaluates the fitness of a genome.
///
/// Implemented automatically for any closure `Fn(&G) -> F`, so you can pass
/// a function directly where a `FitnessEvaluator` is expected.
///
/// # Examples
///
/// ```
/// use evolve::fitness::FitnessEvaluator;
///
/// // A closure works as a fitness evaluator
/// let eval = |g: &[u8; 3]| g.iter().map(|x| *x as u32).sum::<u32>();
/// assert_eq!(eval.evaluate(&[1, 2, 3]), 6);
/// ```
pub trait FitnessEvaluator<G, F> {
    /// Computes the fitness of the given genome.
    fn evaluate(&self, genome: &G) -> F;
}

impl<G, F, E> FitnessEvaluator<G, F> for E
where
    E: Fn(&G) -> F,
{
    fn evaluate(&self, genome: &G) -> F {
        self(genome)
    }
}

/// Compares two fitness values to determine which is better.
///
/// Implemented automatically for any closure `Fn(&F, &F) -> bool`, so you can
/// pass a custom comparison function where a `FitnessComparator` is expected.
///
/// # Examples
///
/// ```
/// use evolve::fitness::FitnessComparator;
///
/// // A closure works as a fitness comparator
/// let cmp = |a: &i32, b: &i32| a > b;
/// assert!(cmp.is_better(&10, &5));
/// ```
pub trait FitnessComparator<F> {
    /// Returns `true` if `f1` is a better fitness than `f2`.
    fn is_better(&self, f1: &F, f2: &F) -> bool;
}

impl<F, C> FitnessComparator<F> for C
where
    C: Fn(&F, &F) -> bool,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        self(f1, f2)
    }
}

/// A [`FitnessComparator`] that treats higher fitness values as better.
///
/// # Examples
///
/// ```
/// use evolve::fitness::{FitnessComparator, Maximize};
///
/// assert!(Maximize.is_better(&10, &5));
/// assert!(!Maximize.is_better(&5, &10));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Maximize;

impl<F> FitnessComparator<F> for Maximize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 > f2
    }
}

/// A [`FitnessComparator`] that treats lower fitness values as better.
///
/// # Examples
///
/// ```
/// use evolve::fitness::{FitnessComparator, Minimize};
///
/// assert!(Minimize.is_better(&5, &10));
/// assert!(!Minimize.is_better(&10, &5));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Minimize;

impl<F> FitnessComparator<F> for Minimize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 < f2
    }
}

// --- GE Fitness ---

use std::marker::PhantomData;

use crate::grammar::grammar_def::GrammarDef;
use crate::grammar::mapper::{map, Codon};
use crate::phenotype::phenotype::PhenotypeBuilder;

/// A fitness evaluator for grammatical evolution.
///
/// Handles codon-to-phenotype mapping internally. The user provides:
/// - A grammar
/// - A builder type (e.g., `BytecodeBuilder<T>`)
/// - An evaluator that scores the phenotype
/// - A penalty for individuals that fail to map
///
/// The user never calls `map()` directly.
pub struct GeFitness<G, C, F, E, B> {
    grammar: G,
    max_wraps: usize,
    evaluator: E,
    penalty: F,
    _marker: PhantomData<(C, B)>,
}

impl<G, C, F, E, B> GeFitness<G, C, F, E, B> {
    /// Creates a GE fitness evaluator.
    ///
    /// - `grammar` — the grammar to map through
    /// - `max_wraps` — maximum codon wraps before declaring invalid
    /// - `evaluator` — receives the phenotype and returns a fitness score
    /// - `penalty` — fitness for individuals that fail to map
    pub fn new(grammar: G, max_wraps: usize, evaluator: E, penalty: F) -> Self {
        Self {
            grammar,
            max_wraps,
            evaluator,
            penalty,
            _marker: PhantomData,
        }
    }
}

impl<G, C, F, E, B> FitnessEvaluator<Vec<C>, F> for GeFitness<G, C, F, E, B>
where
    G: GrammarDef,
    G::Terminal: Clone,
    C: Codon,
    F: Clone,
    B: PhenotypeBuilder<G::Terminal> + Default,
    E: Fn(&B::Output) -> F,
{
    fn evaluate(&self, genome: &Vec<C>) -> F {
        match map(&self.grammar, genome, self.max_wraps, B::default()) {
            Some(phenotype) => (self.evaluator)(&phenotype),
            None => self.penalty.clone(),
        }
    }
}

#[cfg(test)]
mod ge_fitness_tests {
    use super::*;
    use crate::grammar::grammar::Grammar;
    use crate::phenotype::phenotype::{Event, Phenotype};

    #[derive(Debug, PartialEq)]
    struct Program(String);
    impl Phenotype for Program {
        type Input = ();
        type Output = String;
        fn run(&self, _: &()) -> String { self.0.clone() }
    }

    #[derive(Default)]
    struct ProgramBuilder(String);
    impl PhenotypeBuilder<&'static str> for ProgramBuilder {
        type Output = Program;
        fn push(&mut self, event: Event<&'static str>) {
            if let Event::Terminal(val) = event {
                self.0.push_str(val);
            }
        }
        fn finish(self) -> Program { Program(self.0) }
    }

    fn simple_grammar() -> Grammar<&'static str> {
        Grammar::builder()
            .rule("expr", &[&["x"], &["y"]])
            .start("expr")
            .build()
    }

    #[test]
    fn valid_mapping_calls_evaluator() {
        let fitness = GeFitness::<_, u8, f64, _, ProgramBuilder>::new(
            simple_grammar(),
            0,
            |p: &Program| p.run(&()).len() as f64,
            -1.0,
        );
        // codon 0 => "x"
        assert_eq!(fitness.evaluate(&vec![0u8]), 1.0);
    }

    #[test]
    fn invalid_mapping_returns_penalty() {
        // Grammar that requires codons but we give empty
        let g = Grammar::builder()
            .rule("expr", &[&["expr", "op", "expr"], &["x"]])
            .rule("op", &[&["+"], &["-"]])
            .start("expr")
            .build();

        let fitness = GeFitness::<_, u8, f64, _, ProgramBuilder>::new(
            g,
            0,
            |p: &Program| p.run(&()).len() as f64,
            -99.0,
        );
        // codon 0 picks recursive "expr op expr", then wraps exhaust
        assert_eq!(fitness.evaluate(&vec![0u8]), -99.0);
    }
}
