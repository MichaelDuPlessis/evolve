use evolve::fitness::{FitnessComparator, Maximize as RsMaximize, Minimize as RsMinimize};
use pyo3::prelude::*;

/// Python-facing `Maximize` comparator singleton.
#[pyclass(frozen)]
pub struct Maximize;

#[pymethods]
impl Maximize {
    #[new]
    fn new() -> Self { Self }
}

/// Python-facing `Minimize` comparator singleton.
#[pyclass(frozen)]
pub struct Minimize;

#[pymethods]
impl Minimize {
    #[new]
    fn new() -> Self { Self }
}

/// Internal enum used as the concrete `FitnessComparator<f64>` type.
pub enum PyComparator {
    Maximize,
    Minimize,
}

impl FitnessComparator<f64> for PyComparator {
    fn is_better(&self, f1: &f64, f2: &f64) -> bool {
        match self {
            Self::Maximize => RsMaximize.is_better(f1, f2),
            Self::Minimize => RsMinimize.is_better(f1, f2),
        }
    }
}
