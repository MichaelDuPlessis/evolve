//! Phenotype building during grammar mapping.

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

/// An event emitted during grammar mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event<T> {
    /// A terminal value was encountered.
    Terminal(T),
    /// Entering a rule expansion (going deeper in the tree).
    BeginRule,
    /// Finished a rule expansion (coming back up).
    EndRule,
}

/// Trait for building a phenotype from a stream of derivation events.
///
/// The mapper calls `push` for each event during derivation, then
/// calls `finish` to produce the final phenotype.
pub trait PhenotypeBuilder<T> {
    type Output: Phenotype;

    /// Process a derivation event.
    fn push(&mut self, event: Event<T>);

    /// Consume the builder and produce the final phenotype.
    fn finish(self) -> Self::Output;
}
