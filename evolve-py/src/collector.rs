use evolve::{collector::Collector, core::state::State};
use pyo3::prelude::*;

use crate::{comparator::PyComparator, fitness::{PyFitnessCallback, stash_error}};

/// Wraps a Python object as a `Collector`. The Python object may optionally
/// implement `on_start`, `on_generation`, `on_end`, and `finalize`.
///
/// Hook methods are called with `(generation: int, best_fitness: float)`.
/// `finalize()` is called with no args and its return value is returned from `run_with`.
pub struct PyCollectorWrapper {
    obj: Py<PyAny>,
}

impl PyCollectorWrapper {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

fn best_fitness<T>(state: &State<Vec<T>, f64>, fe: &PyFitnessCallback) -> f64
where
    PyFitnessCallback: evolve::fitness::FitnessEvaluator<Vec<T>, f64>,
{
    state
        .population()
        .iter()
        .map(|ind| *ind.fitness(fe))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn call_hook<T>(obj: &Py<PyAny>, method: &str, state: &State<Vec<T>, f64>, fe: &PyFitnessCallback)
where
    PyFitnessCallback: evolve::fitness::FitnessEvaluator<Vec<T>, f64>,
{
    Python::with_gil(|py| {
        let bound = obj.bind(py);
        if let Ok(m) = bound.getattr(method) {
            if m.is_callable() {
                let generation = state.generation();
                let best = best_fitness(state, fe);
                if let Err(e) = m.call1((generation, best)) {
                    stash_error(py, e);
                }
            }
        }
    });
}

// Implement Collector for all 10 dtype Vec<T> combinations via a macro.
macro_rules! impl_collector {
    ($($t:ty),*) => {
        $(
            impl Collector<Vec<$t>, f64, PyFitnessCallback, PyComparator> for PyCollectorWrapper {
                type Result = Py<PyAny>;

                fn on_start(&mut self, state: &State<Vec<$t>, f64>, fe: &PyFitnessCallback, _cmp: &PyComparator) {
                    call_hook(&self.obj, "on_start", state, fe);
                }

                fn on_generation(&mut self, state: &State<Vec<$t>, f64>, fe: &PyFitnessCallback, _cmp: &PyComparator) {
                    call_hook(&self.obj, "on_generation", state, fe);
                }

                fn on_end(&mut self, state: &State<Vec<$t>, f64>, fe: &PyFitnessCallback, _cmp: &PyComparator) {
                    call_hook(&self.obj, "on_end", state, fe);
                }

                fn finalize(self, _state: State<Vec<$t>, f64>) -> Py<PyAny> {
                    Python::with_gil(|py| {
                        let bound = self.obj.bind(py);
                        if let Ok(m) = bound.getattr("finalize") {
                            if m.is_callable() {
                                match m.call0() {
                                    Ok(v) => return v.unbind(),
                                    Err(e) => { stash_error(py, e); }
                                }
                            }
                        }
                        py.None()
                    })
                }
            }
        )*
    };
}

impl_collector!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
