//! Phenotype building during grammar mapping.

use crate::ge::grammar::{Grammar, Symbol};
use crate::ge::grammar_def::GrammarDef;

/// Trait for building a phenotype during grammar mapping.
///
/// The mapper calls these methods as it expands the derivation tree.
/// Implement this to produce ASTs, typed expressions, or any structured
/// output instead of plain strings.
pub trait PhenotypeBuilder<G: GrammarDef> {
    /// The final phenotype type produced.
    type Output;

    /// Called when a terminal symbol is encountered.
    fn terminal(&mut self, symbol: G::Symbol, grammar: &G);

    /// Called when a non-terminal is about to be expanded.
    fn begin_rule(&mut self, symbol: G::Symbol, production_index: usize, grammar: &G);

    /// Called when a non-terminal's expansion is complete.
    fn end_rule(&mut self);

    /// Consume the builder and produce the final phenotype.
    fn finish(self) -> Self::Output;
}

/// Default builder that concatenates terminal values into a [`String`].
///
/// # Examples
///
/// ```
/// use evolve::ge::grammar::Grammar;
/// use evolve::ge::phenotype::StringBuilder;
/// use evolve::ge::mapper::map;
///
/// let grammar = Grammar::builder()
///     .rule("s", &[&["hello", " ", "world"]])
///     .start("s")
///     .build();
///
/// let result = map(&grammar, &[0u8], 0, StringBuilder::new());
/// assert_eq!(result, Some("hello world".to_string()));
/// ```
#[derive(Default)]
pub struct StringBuilder(String);

impl StringBuilder {
    pub fn new() -> Self {
        Self(String::new())
    }
}

impl<T: AsRef<str>> PhenotypeBuilder<Grammar<T>> for StringBuilder {
    type Output = String;

    fn terminal(&mut self, symbol: Symbol, grammar: &Grammar<T>) {
        if let Symbol::Terminal(idx) = symbol {
            self.0.push_str(grammar.terminal_value(idx).as_ref());
        }
    }

    fn begin_rule(&mut self, _: Symbol, _: usize, _: &Grammar<T>) {}

    fn end_rule(&mut self) {}

    fn finish(self) -> String {
        self.0
    }
}
