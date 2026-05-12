//! Phenotype building during grammar mapping.

use crate::ge::grammar_def::GrammarDef;

/// A runnable phenotype produced by grammatical evolution.
///
/// Represents an evolved program that can be executed with an input
/// to produce an output.
pub trait Phenotype {
    /// The input type the program accepts.
    type Input;
    /// The output type the program produces.
    type Output;

    /// Runs the program with the given input.
    fn run(&self, input: &Self::Input) -> Self::Output;
}

/// Trait for building a phenotype during grammar mapping.
///
/// The mapper calls these methods as it expands the derivation tree.
/// Implement this to produce ASTs, typed expressions, or any structured
/// output instead of plain strings.
pub trait PhenotypeBuilder<G: GrammarDef> {
    /// The final phenotype type produced.
    type Output: Phenotype;

    /// Called when a terminal symbol is encountered.
    fn terminal(&mut self, symbol: G::Symbol, grammar: &G);

    /// Called when a non-terminal is about to be expanded.
    fn begin_rule(&mut self, symbol: G::Symbol, production_index: usize, grammar: &G);

    /// Called when a non-terminal's expansion is complete.
    fn end_rule(&mut self);

    /// Consume the builder and produce the final phenotype.
    fn finish(self) -> Self::Output;
}
