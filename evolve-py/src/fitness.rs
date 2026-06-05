use evolve::fitness::FitnessEvaluator;
use pyo3::prelude::*;

/// Wraps a Python callable as a `FitnessEvaluator<Vec<u8>, f64>`.
///
/// The Python callable receives a list of ints (the genome) and must return a float.
pub struct PyFitnessCallback {
    func: Py<PyAny>,
}

impl PyFitnessCallback {
    pub fn new(func: Py<PyAny>) -> Self {
        Self { func }
    }
}

impl FitnessEvaluator<Vec<u8>, f64> for PyFitnessCallback {
    fn evaluate(&self, genome: &Vec<u8>) -> f64 {
        Python::with_gil(|py| {
            let py_genome: Vec<u8> = genome.clone();
            self.func
                .bind(py)
                .call1((py_genome,))
                .expect("fitness function raised an exception")
                .extract::<f64>()
                .expect("fitness function must return a float")
        })
    }
}
