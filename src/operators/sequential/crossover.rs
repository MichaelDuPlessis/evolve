//! Crossover operators.
//!
//! - [`SinglePoint`] — single-point crossover
//! - [`Uniform`] — uniform (per-gene) crossover
//! - [`TwoPoint`] — two-point crossover
//! - [`Arithmetic`] — arithmetic blending for continuous genomes

/// Arithmetic crossover for continuous-valued genomes.
pub mod arithmetic;
/// Single-point crossover operator.
pub mod single_point;
/// Two-point crossover operator.
pub mod two_point;
/// Uniform crossover operator.
pub mod uniform;

pub use arithmetic::Arithmetic;
pub use single_point::SinglePoint;
pub use two_point::TwoPoint;
pub use uniform::Uniform;
