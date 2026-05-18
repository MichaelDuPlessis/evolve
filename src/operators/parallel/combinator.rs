//! Parallel operator combinators.
//!
//! - [`Combine`] — runs operators on the same input in parallel and merges outputs
//! - [`Fill`] — fills a population in parallel
//! - [`Proportional`] — splits output proportionally across operators in parallel
//! - [`Repeat`] — applies an operator N times in parallel

mod combine;
mod fill;
mod proportional;
mod repeat;

pub use combine::Combine;
pub use fill::Fill;
pub use proportional::Proportional;
pub use repeat::Repeat;
