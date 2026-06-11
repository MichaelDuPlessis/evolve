use evolve::initialization::RangedRandom;
use pyo3::prelude::*;

/// Python-facing initializer that generates variable-length `Vec<u8>` genomes.
#[pyclass(name = "RangedRandom")]
pub struct PyRangedRandom {
    pub inner: RangedRandom<u8>,
}

#[pymethods]
impl PyRangedRandom {
    #[new]
    fn new(min_len: usize, max_len: usize) -> Self {
        Self {
            inner: RangedRandom::new(min_len..max_len + 1),
        }
    }
}
