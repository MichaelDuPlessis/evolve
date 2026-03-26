//! Composable operator combinators.
//!
//! Combinators wrap one or more [`GeneticOperator`](crate::operators::sequential::GeneticOperator)s
//! to structure the flow of the algorithm:
//!
//! - [`Pipeline`] — chains operators sequentially
//! - [`Combine`] — runs operators on the same input and merges outputs
//! - [`Weighted`] — probabilistically selects one operator per invocation
//! - [`Repeat`] — applies an operator N times
//! - [`Fill`] — repeats an operator until a target population size is reached

mod combine;
mod fill;
mod pipeline;
mod repeat;
mod weighted;

pub use combine::Combine;
pub use fill::Fill;
pub use pipeline::Pipeline;
pub use repeat::Repeat;
pub use weighted::Weighted;
