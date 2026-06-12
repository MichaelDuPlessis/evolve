use evolve::{core::state::State, termination::{MaxGenerations, TerminationCondition}};
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

/// Internal termination: either MaxGenerations or a Python callable.
///
/// The Python callable receives `(generation: int, best_fitness: float)` and
/// must return `True` to stop.
pub enum PyTermination {
    MaxGenerations(MaxGenerations),
    PythonCallback(Py<PyAny>),
}

impl<G: Clone> TerminationCondition<G, f64> for PyTermination {
    fn should_terminate(&self, state: &State<G, f64>) -> bool {
        match self {
            Self::MaxGenerations(t) => t.should_terminate(state),
            Self::PythonCallback(cb) => {
                let generation = state.generation();
                // Best fitness: find already-computed fitness in population.
                // After the first generation operators run, fitness is lazily cached.
                let best: f64 = state
                    .population()
                    .iter()
                    .filter_map(|ind| ind.try_fitness().copied())
                    .fold(f64::NEG_INFINITY, f64::max);
                Python::with_gil(|py| {
                    cb.bind(py)
                        .call1((generation, best))
                        .expect("termination callable raised an exception")
                        .extract::<bool>()
                        .expect("termination callable must return a bool")
                })
            }
        }
    }
}

/// Parse a Python object into a `PyTermination`.
pub fn extract_termination(obj: &Bound<'_, PyAny>) -> PyResult<PyTermination> {
    if let Ok(cell) = obj.downcast::<PyMaxGenerations>() {
        return Ok(PyTermination::MaxGenerations(cell.borrow().inner));
    }
    if obj.is_callable() {
        return Ok(PyTermination::PythonCallback(obj.clone().unbind()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "termination must be MaxGenerations() or a callable(generation, best_fitness) -> bool",
    ))
}
