//! Parallel genetic operators.
//!
//! Requires the `parallel` feature to be enabled.
//!
//! - [`combinator`] — parallel combinators (Fill, Combine, Repeat)
//! - [`crossover`] — parallel crossover operators
//! - [`mutation`] — parallel mutation operators

pub mod combinator;
pub mod crossover;
pub mod mutation;
