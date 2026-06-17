use evolve::fitness::FitnessEvaluator;
use pyo3::prelude::*;

/// Wraps a Python callable as a `FitnessEvaluator<Vec<T>, f64>` for any genome element type T.
///
/// The Python callable receives a list and must return a float.
pub struct PyFitnessCallback {
    pub func: Py<PyAny>,
}

impl PyFitnessCallback {
    pub fn new(func: Py<PyAny>) -> Self {
        Self { func }
    }
}

/// Macro: implement FitnessEvaluator<Vec<$t>, f64> for PyFitnessCallback.
macro_rules! impl_fitness {
    ($($t:ty),*) => {
        $(
            impl FitnessEvaluator<Vec<$t>, f64> for PyFitnessCallback {
                fn evaluate(&self, genome: &Vec<$t>) -> f64 {
                    Python::with_gil(|py| {
                        let py_genome = match pyo3::types::PyList::new(
                            py,
                            genome.iter().map(|&v| v.into_pyobject(py).unwrap()),
                        ) {
                            Ok(list) => list,
                            Err(e) => { stash_error(py, e); return f64::NAN; }
                        };
                        let result = match self.func.bind(py).call1((py_genome,)) {
                            Ok(r) => r,
                            Err(e) => { stash_error(py, e); return f64::NAN; }
                        };
                        match result.extract::<f64>() {
                            Ok(v) => v,
                            Err(_) => {
                                stash_error(py, pyo3::exceptions::PyTypeError::new_err(
                                    "fitness function must return a float",
                                ));
                                f64::NAN
                            }
                        }
                    })
                }
            }
        )*
    };
}

impl_fitness!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// ── Error stash ──────────────────────────────────────────────────────────────

thread_local! {
    static PENDING_ERROR: std::cell::Cell<Option<PyErr>> = const { std::cell::Cell::new(None) };
}

/// Store a Python error so it can be re-raised after the EA run returns.
pub fn stash_error(_py: Python<'_>, err: PyErr) {
    PENDING_ERROR.with(|cell| {
        // Take existing (drop it) and store the new one
        let _ = cell.replace(Some(err));
    });
}

/// If a Python error was stashed during an EA run, take and return it.
pub fn take_stashed_error() -> Option<PyErr> {
    PENDING_ERROR.with(|cell| cell.replace(None))
}
