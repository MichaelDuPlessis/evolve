/// Parallel creep mutation.
pub mod creep;
/// Parallel Gaussian mutation.
pub mod gaussian;
/// Parallel segment inversion mutation.
pub mod inversion;
/// Parallel random reset mutation.
pub mod random_reset;
/// Parallel swap mutation.
pub mod swap;

pub use creep::Creep;
pub use gaussian::Gaussian;
pub use inversion::Inversion;
pub use random_reset::RandomReset;
pub use swap::Swap;

/// Helper trait — see sequential mutation module.
pub(crate) trait GeneCollection {}
impl<T> GeneCollection for Vec<T> {}
impl<T> GeneCollection for Box<[T]> {}
impl<T> GeneCollection for [T] {}
impl<T, const N: usize> GeneCollection for [T; N] {}
