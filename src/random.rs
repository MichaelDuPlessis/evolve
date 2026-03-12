use rand::{Rng, RngExt};

/// Implementing this trait means that a object can be created using a random number generator.
pub trait Randomizable<R>
where
    R: Rng,
{
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

// Apply macro to common numeric types
impl_randomizable_for_numbers!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, char
);
