use evolve::termination::MaxGenerations;
use pyo3::prelude::*;

/// Stop the algorithm after a fixed number of generations.
#[pyclass(name = "MaxGenerations")]
pub struct PyMaxGenerations {
    pub inner: MaxGenerations,
}

#[pymethods]
impl PyMaxGenerations {
    /// Create a termination condition that stops after `generations` generations.
    #[new]
    fn new(generations: usize) -> Self {
        Self {
            inner: MaxGenerations::new(generations),
        }
    }
}
