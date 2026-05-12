//! Phenotype building during grammar mapping.

pub mod bytecode;

/// Events emitted during grammar derivation, consumed by a [`PhenotypeBuilder`].
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
///
/// # Examples
///
/// ```
/// use evolve::phenotype::{Event, PhenotypeBuilder};
///
/// struct Count(usize);
///
/// #[derive(Default)]
/// struct Counter(usize);
/// impl PhenotypeBuilder<&'static str> for Counter {
///     type Output = Count;
///     fn push(&mut self, event: Event<&str>) {
///         if matches!(event, Event::Terminal(_)) { self.0 += 1; }
///     }
///     fn finish(self) -> Count { Count(self.0) }
/// }
/// ```
pub trait PhenotypeBuilder<T> {
    /// The type produced by the builder.
    type Output;

    /// Process a derivation event.
    fn push(&mut self, event: Event<T>);

    /// Consume the builder and produce the final phenotype.
    fn finish(self) -> Self::Output;
}
