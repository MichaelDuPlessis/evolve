//! GE fitness evaluator wrapper.

use crate::fitness::FitnessEvaluator;
use crate::ge::grammar_def::GrammarDef;
use crate::ge::mapper::{Codon, map};
use crate::ge::phenotype::PhenotypeBuilder;
use std::marker::PhantomData;

/// A fitness evaluator for grammatical evolution.
///
/// Maps codon genomes through a grammar, then evaluates the phenotype.
/// Returns a penalty fitness for individuals that fail to map.
///
/// # Examples
///
/// ```
/// use evolve::ge::fitness::GeFitness;
/// use evolve::ge::grammar::Grammar;
/// use evolve::ge::phenotype::StringBuilder;
/// use evolve::fitness::FitnessEvaluator;
///
/// let grammar = Grammar::builder()
///     .rule("s", &[&["a"], &["b"]])
///     .start("s")
///     .build();
///
/// let ge = GeFitness::<_, u8, _, _, StringBuilder>::new(
///     grammar,
///     3,
///     |p: &String| if p == "a" { 1.0 } else { 0.0 },
///     -1.0,
/// );
///
/// assert_eq!(ge.evaluate(&vec![0u8]), 1.0);
/// assert_eq!(ge.evaluate(&vec![1u8]), 0.0);
/// ```
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
    /// - `evaluator` — scores the phenotype
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
    C: Codon,
    F: Clone,
    B: PhenotypeBuilder<G> + Default,
    E: Fn(&B::Output) -> F,
{
    fn evaluate(&self, genome: &Vec<C>) -> F {
        let builder = B::default();
        match map(&self.grammar, genome, self.max_wraps, builder) {
            Some(phenotype) => (self.evaluator)(&phenotype),
            None => self.penalty.clone(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ge::grammar::{Grammar, Symbol};
    use crate::ge::phenotype::StringBuilder;

    #[test]
    fn valid_mapping_returns_evaluator_result() {
        let grammar = Grammar::builder()
            .rule("s", &[&["a"], &["b"]])
            .start("s")
            .build();

        let ge = GeFitness::<_, u8, f64, _, StringBuilder>::new(
            grammar,
            3,
            |p: &String| if p == "a" { 1.0 } else { 0.0 },
            -1.0,
        );

        // codon 0 % 2 = 0 → "a"
        assert_eq!(ge.evaluate(&vec![0u8]), 1.0);
    }

    #[test]
    fn invalid_mapping_returns_penalty() {
        let grammar = Grammar::builder()
            .rule("s", &[&["s", "s"], &["x"]])
            .start("s")
            .build();

        let ge =
            GeFitness::<_, u8, f64, _, StringBuilder>::new(grammar, 0, |_: &String| 1.0, -999.0);

        // codon 0 % 2 = 0 → recursive, with max_wraps=0 should fail
        assert_eq!(ge.evaluate(&vec![0u8]), -999.0);
    }

    #[test]
    fn custom_builder() {
        #[derive(Default)]
        struct CountBuilder(usize);
        impl PhenotypeBuilder<Grammar> for CountBuilder {
            type Output = usize;
            fn terminal(&mut self, _: Symbol, _: &Grammar) {
                self.0 += 1;
            }
            fn begin_rule(&mut self, _: Symbol, _: usize, _: &Grammar) {}
            fn end_rule(&mut self) {}
            fn finish(self) -> usize {
                self.0
            }
        }

        let grammar = Grammar::builder()
            .rule("s", &[&["hello", " ", "world"], &["hi"]])
            .start("s")
            .build();

        let ge = GeFitness::<_, u8, f64, _, CountBuilder>::new(
            grammar,
            3,
            |count: &usize| *count as f64,
            0.0,
        );

        // codon 0 % 2 = 0 → production with 3 terminals
        assert_eq!(ge.evaluate(&vec![0u8]), 3.0);
    }
}
