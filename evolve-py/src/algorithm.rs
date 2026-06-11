use std::num::NonZero;

use evolve::{algorithm::EvolutionaryAlgorithm, initialization::RangedRandom, termination::MaxGenerations};
use pyo3::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    comparator::{Maximize, Minimize, PyComparator},
    fitness::PyFitnessCallback,
    initializer::PyRangedRandom,
    operators::{PyOperatorU8, extract_op},
    result::{PyIndividual, PyRunResult},
    termination::PyMaxGenerations,
};

type EaU8 = EvolutionaryAlgorithm<
    Vec<u8>,
    f64,
    RangedRandom<u8>,
    MaxGenerations,
    PyFitnessCallback,
    PyOperatorU8,
    SmallRng,
    PyComparator,
>;

/// Evolutionary algorithm runner. Compose with an initializer, operators, fitness function, and termination condition.
#[pyclass(name = "EvolutionaryAlgorithm")]
pub struct PyEvolutionaryAlgorithm {
    inner: EaU8,
    /// Kept separately so we can evaluate fitness on the final population.
    fitness_evaluator: PyFitnessCallback,
    comparator: PyComparator,
}

#[pymethods]
impl PyEvolutionaryAlgorithm {
    /// Create a new `EvolutionaryAlgorithm`. `comparator` defaults to `Maximize`; `seed` defaults to a random seed.
    #[new]
    #[pyo3(signature = (initializer, operators, fitness, termination, population_size, comparator=None, seed=None))]
    fn new(
        initializer: &PyRangedRandom,
        operators: &Bound<'_, PyAny>,
        fitness: Py<PyAny>,
        termination: &PyMaxGenerations,
        population_size: usize,
        comparator: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let pop_size = NonZero::new(population_size)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("population_size must be > 0"))?;

        let op = extract_op(operators)?;

        let cmp = match comparator {
            None => PyComparator::Maximize,
            Some(obj) if obj.downcast::<Maximize>().is_ok() => PyComparator::Maximize,
            Some(obj) if obj.downcast::<Minimize>().is_ok() => PyComparator::Minimize,
            Some(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "comparator must be Maximize() or Minimize()",
                ));
            }
        };

        let cmp_clone = match &cmp {
            PyComparator::Maximize => PyComparator::Maximize,
            PyComparator::Minimize => PyComparator::Minimize,
        };

        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_rng(&mut rand::rng()),
        };

        // Clone the callable so we can re-use it after run()
        let fe_clone = Python::with_gil(|py| fitness.clone_ref(py));
        let fe1 = PyFitnessCallback::new(fitness);
        let fe2 = PyFitnessCallback::new(fe_clone);

        let ea = EvolutionaryAlgorithm::new(
            initializer.inner.clone(),
            termination.inner,
            fe1,
            op,
            pop_size,
            rng,
            cmp,
        );

        Ok(Self {
            inner: ea,
            fitness_evaluator: fe2,
            comparator: cmp_clone,
        })
    }

    /// Run the algorithm to termination and return a `RunResult` with the final population and statistics.
    fn run(&mut self, py: Python<'_>) -> PyResult<PyRunResult> {
        let run_result = py.allow_threads(|| self.inner.run());

        let fe = &self.fitness_evaluator;

        let population: Vec<PyIndividual> = run_result
            .population()
            .iter()
            .map(|ind| {
                let fitness = *ind.fitness(fe);
                PyIndividual::new(ind.genome().clone(), fitness)
            })
            .collect();

        let best = {
            use evolve::fitness::FitnessComparator;
            run_result
                .population()
                .iter()
                .map(|ind| {
                    let fitness = *ind.fitness(fe);
                    (ind.genome().clone(), fitness)
                })
                .reduce(|a, b| if self.comparator.is_better(&a.1, &b.1) { a } else { b })
                .map(|(genome, fitness)| PyIndividual::new(genome, fitness))
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("population is empty"))?
        };

        let best_fitness: Vec<f64> = run_result.best_fitness().to_vec();
        let generation_durations: Vec<f64> = run_result
            .generation_durations()
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        Ok(PyRunResult::new(
            population,
            best,
            run_result.generations(),
            run_result.total_duration().as_secs_f64(),
            best_fitness,
            generation_durations,
        ))
    }
}
