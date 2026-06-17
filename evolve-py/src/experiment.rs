use pyo3::prelude::*;

#[pyclass(name = "Experiment")]
pub struct PyExperiment {
    factory: Py<PyAny>,
    trials: usize,
}

#[pymethods]
impl PyExperiment {
    #[new]
    fn new(factory: Py<PyAny>, trials: usize) -> Self {
        Self { factory, trials }
    }

    fn run(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut results = Vec::with_capacity(self.trials);
        for _ in 0..self.trials {
            let ea_obj = self.factory.call0(py)?;
            let result = ea_obj.call_method0(py, "run")?;
            results.push(result);
        }
        Ok(results)
    }
}
