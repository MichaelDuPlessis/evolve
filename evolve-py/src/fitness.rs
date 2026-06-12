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
                        let py_genome = pyo3::types::PyList::new(
                            py,
                            genome.iter().map(|&v| v.into_pyobject(py).unwrap()),
                        ).unwrap();
                        self.func
                            .bind(py)
                            .call1((py_genome,))
                            .expect("fitness function raised an exception")
                            .extract::<f64>()
                            .expect("fitness function must return a float")
                    })
                }
            }
        )*
    };
}

impl_fitness!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
