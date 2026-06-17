use evolve::fitness::{FitnessComparator, Maximize as RsMaximize, Minimize as RsMinimize};
use pyo3::prelude::*;

/// Comparator that treats higher fitness as better (default).
#[pyclass(frozen)]
pub struct Maximize;

#[pymethods]
impl Maximize {
    /// Create a Maximize comparator.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Comparator that treats lower fitness as better.
#[pyclass(frozen)]
pub struct Minimize;

#[pymethods]
impl Minimize {
    /// Create a Minimize comparator.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Internal enum used as the concrete `FitnessComparator<f64>` type.
pub enum PyComparator {
    Maximize,
    Minimize,
    PythonCallback(Py<PyAny>),
}

impl Clone for PyComparator {
    fn clone(&self) -> Self {
        match self {
            Self::Maximize => Self::Maximize,
            Self::Minimize => Self::Minimize,
            Self::PythonCallback(cb) => {
                Self::PythonCallback(Python::with_gil(|py| cb.clone_ref(py)))
            }
        }
    }
}

impl FitnessComparator<f64> for PyComparator {
    fn is_better(&self, f1: &f64, f2: &f64) -> bool {
        match self {
            Self::Maximize => RsMaximize.is_better(f1, f2),
            Self::Minimize => RsMinimize.is_better(f1, f2),
            Self::PythonCallback(cb) => Python::with_gil(|py| {
                match cb.bind(py).call1((*f1, *f2)) {
                    Ok(v) => v.extract::<bool>().unwrap_or(false),
                    Err(_) => false,
                }
            }),
        }
    }
}
