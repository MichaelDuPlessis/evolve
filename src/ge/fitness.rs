//! GE fitness evaluator — maps genomes to phenotypes and scores them.

use std::marker::PhantomData;

use crate::fitness::FitnessEvaluator;
use crate::ge::grammar_def::GrammarDef;
use crate::ge::mapper::{map, Codon};
use crate::ge::phenotype::PhenotypeBuilder;

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
mod tests {
    use super::*;
    use crate::ge::grammar::Grammar;
    use crate::ge::phenotype::{Event, Phenotype};

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
