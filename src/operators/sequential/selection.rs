//! Selection operators.
//!
//! - [`Tournament`] — tournament selection with replacement
//! - [`Elitism`] — preserve the best N individuals
//! - [`RouletteWheel`] — fitness-proportionate selection
//! - [`Rank`] — rank-based selection
//! - [`Sus`] — stochastic universal sampling

/// Elitism selection — preserve the best N individuals.
mod elitism;
/// Rank-based selection.
mod rank;
/// Fitness-proportionate (roulette wheel) selection.
mod roulette;
/// Stochastic universal sampling.
mod sus;
/// Tournament selection with replacement.
mod tournament;

pub use elitism::Elitism;
pub use rank::Rank;
pub use roulette::RouletteWheel;
pub use sus::Sus;
pub use tournament::Tournament;
