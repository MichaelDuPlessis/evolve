use evolve::initialization::RangedRandom;
use pyo3::prelude::*;

/// Initialize genomes with random u8 values and a random length in `[min_len, max_len]`.
#[pyclass(name = "RangedRandom")]
pub struct PyRangedRandom {
    pub inner: RangedRandom<u8>,
}

#[pymethods]
impl PyRangedRandom {
    /// Create an initializer producing genomes of random length between `min_len` and `max_len` (inclusive).
    #[new]
    fn new(min_len: usize, max_len: usize) -> Self {
        Self {
            inner: RangedRandom::new(min_len..max_len + 1),
        }
    }
}
