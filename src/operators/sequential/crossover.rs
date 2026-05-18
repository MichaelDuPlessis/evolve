//! Crossover operators.
//!
//! - [`SinglePoint`] — single-point crossover
//! - [`Uniform`] — uniform (per-gene) crossover
//! - [`TwoPoint`] — two-point crossover
//! - [`Arithmetic`] — arithmetic blending for continuous genomes

/// Single-point crossover operator.
pub mod single_point;
/// Uniform crossover operator.
pub mod uniform;
/// Two-point crossover operator.
pub mod two_point;
/// Arithmetic crossover for continuous-valued genomes.
pub mod arithmetic;

pub use single_point::SinglePoint;
pub use uniform::Uniform;
pub use two_point::TwoPoint;
pub use arithmetic::Arithmetic;
