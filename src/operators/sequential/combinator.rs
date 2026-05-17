//! Composable operator combinators.
//!
//! Combinators wrap one or more [`GeneticOperator`](crate::operators::GeneticOperator)s
//! to structure the flow of the algorithm:
//!
//! - [`Pipeline`] — chains operators sequentially
//! - [`Combine`] — runs operators on the same input and merges outputs
//! - [`Weighted`] — probabilistically selects one operator per invocation
//! - [`Repeat`] — applies an operator N times
//! - [`Fill`] — repeats an operator until a target population size is reached
//! - [`Proportional`] — splits output proportionally across multiple operators

mod combine;
pub mod fill;
mod pipeline;
mod proportional;
mod repeat;
mod weighted;

pub use combine::Combine;
pub use fill::Fill;
pub use pipeline::Pipeline;
pub use proportional::Proportional;
pub use repeat::Repeat;
pub use weighted::Weighted;
