use evolve::termination::MaxGenerations;
use pyo3::prelude::*;

/// Python-facing termination condition: stop after N generations.
#[pyclass(name = "MaxGenerations")]
pub struct PyMaxGenerations {
    pub inner: MaxGenerations,
}

#[pymethods]
impl PyMaxGenerations {
    #[new]
    fn new(generations: usize) -> Self {
        Self {
            inner: MaxGenerations::new(generations),
        }
    }
}
