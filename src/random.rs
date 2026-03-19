//! Random value generation for genomes and genes.
//!
//! Defines the [`Randomizable`] trait, which allows types to be created from
//! a random number generator. Used by initializers and mutation operators.

use rand::{Rng, RngExt};
use std::array;

/// A type that can be created from a random number generator.
///
/// Used by operators and initializers to generate random genomes and gene values.
///
/// # Examples
///
/// ```
/// use evolve::random::Randomizable;
///
/// let mut rng = rand::rng();
/// let value: u32 = Randomizable::random(&mut rng);
/// let array: [u8; 4] = Randomizable::random(&mut rng);
/// ```
pub trait Randomizable<R>
where
    R: Rng,
{
    /// Generates a random instance of this type.
    fn random(rng: &mut R) -> Self;
}

macro_rules! impl_randomizable_for_numbers {
    ($($t:ty),*) => {
        $(
            impl<R> Randomizable<R> for $t
            where
                R: Rng,
            {
                fn random(rng: &mut R) -> Self {
                    rng.random::<$t>()
                }
            }
        )*
    };
}

impl_randomizable_for_numbers!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, char
);

// Be able to create random arrays with random elements
impl<R, T, const N: usize> Randomizable<R> for [T; N]
where
    R: Rng,
    T: Randomizable<R>,
{
    fn random(rng: &mut R) -> Self {
        array::from_fn(|_| T::random(rng))
    }
}
