use pyo3::prelude::*;

/// A single evaluated individual. Access `genome` (list of ints) and `fitness` (float).
#[pyclass(name = "Individual")]
pub struct PyIndividual {
    genome: Vec<u8>,
    fitness: f64,
}

#[pymethods]
impl PyIndividual {
    #[getter]
    fn genome(&self) -> Vec<u32> {
        self.genome.iter().map(|&b| b as u32).collect()
    }

    #[getter]
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

impl PyIndividual {
    pub fn new(genome: Vec<u8>, fitness: f64) -> Self {
        Self { genome, fitness }
    }
}

/// The result of an EA run. Contains the final population, best individual, and per-generation statistics.
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
    fn population(&self) -> Vec<PyIndividual> {
        self.population
            .iter()
            .map(|ind| PyIndividual::new(ind.genome.clone(), ind.fitness))
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

    /// Return the best individual found across all generations.
    fn best(&self) -> PyIndividual {
        PyIndividual::new(self.best_individual.genome.clone(), self.best_individual.fitness)
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
