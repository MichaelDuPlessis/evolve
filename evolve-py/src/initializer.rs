use pyo3::prelude::*;

use crate::types::Dtype;

/// Initialize genomes with random values and a random length in `[min_len, max_len]`.
#[pyclass(name = "RangedRandom")]
pub struct PyRangedRandom {
    pub min_len: usize,
    pub max_len: usize,
    pub dtype: Dtype,
}

#[pymethods]
impl PyRangedRandom {
    #[new]
    #[pyo3(signature = (min_len, max_len, dtype="u8"))]
    fn new(min_len: usize, max_len: usize, dtype: &str) -> PyResult<Self> {
        Ok(Self {
            min_len,
            max_len,
            dtype: Dtype::parse(dtype)?,
        })
    }
}

/// Initialize genomes with random values and a fixed genome length.
/// Internally uses `RangedRandom` with equal min/max lengths.
#[pyclass(name = "Random")]
pub struct PyRandom {
    pub genome_length: usize,
    pub dtype: Dtype,
}

#[pymethods]
impl PyRandom {
    #[new]
    #[pyo3(signature = (genome_length, dtype="u8"))]
    fn new(genome_length: usize, dtype: &str) -> PyResult<Self> {
        Ok(Self {
            genome_length,
            dtype: Dtype::parse(dtype)?,
        })
    }
}
