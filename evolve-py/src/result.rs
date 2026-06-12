use std::fmt::Debug;
use pyo3::prelude::*;

/// A single evaluated individual. Access `genome` (list) and `fitness` (float).
#[pyclass(name = "Individual")]
pub struct PyIndividual {
    genome: PyObject,
    fitness: f64,
}

#[pymethods]
impl PyIndividual {
    #[getter]
    fn genome(&self, py: Python<'_>) -> PyObject {
        self.genome.clone_ref(py)
    }

    #[getter]
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

impl PyIndividual {
    pub fn new(genome: PyObject, fitness: f64) -> Self {
        Self { genome, fitness }
    }
}

/// Convert a `&[T]` genome slice to a Python list object.
pub fn genome_to_pyobject<T>(py: Python<'_>, genome: &[T]) -> PyObject
where
    for<'py> T: IntoPyObject<'py> + Copy,
    for<'py> <T as IntoPyObject<'py>>::Error: Debug,
{
    use pyo3::BoundObject;
    let items: Vec<PyObject> = genome
        .iter()
        .map(|&v| v.into_pyobject(py).unwrap().into_any().unbind())
        .collect();
    pyo3::types::PyList::new(py, items).unwrap().into()
}

/// The result of an EA run.
#[pyclass(name = "RunResult")]
pub struct PyRunResult {
    population: Vec<PyIndividual>,
    best_individual: PyIndividual,
    generations: usize,
    total_duration_secs: f64,
    best_fitness: Vec<f64>,
    generation_durations_secs: Vec<f64>,
}

#[pymethods]
impl PyRunResult {
    #[getter]
    fn population(&self, py: Python<'_>) -> Vec<PyIndividual> {
        self.population
            .iter()
            .map(|ind| PyIndividual::new(ind.genome.clone_ref(py), ind.fitness))
            .collect()
    }

    #[getter]
    fn generations(&self) -> usize {
        self.generations
    }

    #[getter]
    fn total_duration(&self) -> f64 {
        self.total_duration_secs
    }

    #[getter]
    fn best_fitness(&self) -> Vec<f64> {
        self.best_fitness.clone()
    }

    #[getter]
    fn generation_durations(&self) -> Vec<f64> {
        self.generation_durations_secs.clone()
    }

    fn best(&self, py: Python<'_>) -> PyIndividual {
        PyIndividual::new(self.best_individual.genome.clone_ref(py), self.best_individual.fitness)
    }
}

impl PyRunResult {
    pub fn new(
        population: Vec<PyIndividual>,
        best_individual: PyIndividual,
        generations: usize,
        total_duration_secs: f64,
        best_fitness: Vec<f64>,
        generation_durations_secs: Vec<f64>,
    ) -> Self {
        Self {
            population,
            best_individual,
            generations,
            total_duration_secs,
            best_fitness,
            generation_durations_secs,
        }
    }
}
