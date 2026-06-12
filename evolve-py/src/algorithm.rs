use std::num::NonZero;

use evolve::{
    algorithm::EvolutionaryAlgorithm,
    initialization::RangedRandom,
};
use pyo3::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    collector::PyCollectorWrapper,
    comparator::{Maximize, Minimize, PyComparator},
    fitness::PyFitnessCallback,
    initializer::{PyRangedRandom, PyRandom},
    operators::{
        extract_op_f32, extract_op_f64, extract_op_i16, extract_op_i32, extract_op_i64,
        extract_op_i8, extract_op_u16, extract_op_u32, extract_op_u64, extract_op_u8,
    },
    result::{PyIndividual, PyRunResult, genome_to_pyobject},
    termination::{extract_termination, PyTermination},
    types::{Dtype, EaInner, PyInitializer},
};

/// Evolutionary algorithm runner.
#[pyclass(name = "EvolutionaryAlgorithm")]
pub struct PyEvolutionaryAlgorithm {
    inner: EaInner,
    fitness_evaluator: PyFitnessCallback,
    comparator: PyComparator,
}

#[pymethods]
impl PyEvolutionaryAlgorithm {
    #[new]
    #[pyo3(signature = (initializer, operators, fitness, termination, population_size, comparator=None, seed=None))]
    fn new(
        initializer: &Bound<'_, PyAny>,
        operators: &Bound<'_, PyAny>,
        fitness: Py<PyAny>,
        termination: &Bound<'_, PyAny>,
        population_size: usize,
        comparator: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let pop_size = NonZero::new(population_size)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("population_size must be > 0"))?;

        let cmp = match comparator {
            None => PyComparator::Maximize,
            Some(obj) if obj.downcast::<Maximize>().is_ok() => PyComparator::Maximize,
            Some(obj) if obj.downcast::<Minimize>().is_ok() => PyComparator::Minimize,
            Some(obj) if obj.is_callable() => PyComparator::PythonCallback(obj.clone().unbind()),
            Some(_) => return Err(pyo3::exceptions::PyTypeError::new_err(
                "comparator must be Maximize(), Minimize(), or a callable(a, b) -> bool",
            )),
        };

        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_rng(&mut rand::rng()),
        };

        let fe_clone = Python::with_gil(|py| fitness.clone_ref(py));
        let fe1 = PyFitnessCallback::new(fitness);
        let fe2 = PyFitnessCallback::new(fe_clone);
        let cmp_clone = cmp.clone();

        let term = extract_termination(termination)?;

        // Extract dtype and initializer from the initializer argument
        let (dtype, py_initializer) = extract_initializer(initializer)?;

        let inner = build_ea_inner(operators, fe1, term, pop_size, rng, cmp, dtype, py_initializer)?;
        Ok(Self { inner, fitness_evaluator: fe2, comparator: cmp_clone })
    }

    fn run(&mut self, py: Python<'_>) -> PyResult<PyRunResult> {
        let fe = &self.fitness_evaluator;
        let cmp = &self.comparator;
        match &mut self.inner {
            EaInner::U8(ea)  => { let r = py.allow_threads(|| ea.run()); convert_result::<u8>(py, r, fe, cmp) }
            EaInner::U16(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<u16>(py, r, fe, cmp) }
            EaInner::U32(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<u32>(py, r, fe, cmp) }
            EaInner::U64(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<u64>(py, r, fe, cmp) }
            EaInner::I8(ea)  => { let r = py.allow_threads(|| ea.run()); convert_result::<i8>(py, r, fe, cmp) }
            EaInner::I16(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<i16>(py, r, fe, cmp) }
            EaInner::I32(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<i32>(py, r, fe, cmp) }
            EaInner::I64(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<i64>(py, r, fe, cmp) }
            EaInner::F32(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<f32>(py, r, fe, cmp) }
            EaInner::F64(ea) => { let r = py.allow_threads(|| ea.run()); convert_result::<f64>(py, r, fe, cmp) }
        }
    }

    fn run_with(&mut self, py: Python<'_>, collector: Py<PyAny>) -> PyResult<PyObject> {
        let wrapper = PyCollectorWrapper::new(collector.clone_ref(py));
        let result: Py<PyAny> = match &mut self.inner {
            EaInner::U8(ea)  => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::U16(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::U32(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::U64(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::I8(ea)  => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::I16(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::I32(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::I64(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::F32(ea) => py.allow_threads(|| ea.run_with(wrapper)),
            EaInner::F64(ea) => py.allow_threads(|| ea.run_with(wrapper)),
        };
        Ok(result.into_pyobject(py)?.into())
    }
}

/// Enum to pass initializer data through to build_ea_inner without monomorphizing early.
enum InitializerData {
    RangedRandom { min_len: usize, max_len: usize },
    PythonCallback(Py<PyAny>),
}

fn extract_initializer(initializer: &Bound<'_, PyAny>) -> PyResult<(Dtype, InitializerData)> {
    if let Ok(cell) = initializer.downcast::<PyRangedRandom>() {
        let init = cell.borrow();
        return Ok((init.dtype, InitializerData::RangedRandom { min_len: init.min_len, max_len: init.max_len }));
    }
    if let Ok(cell) = initializer.downcast::<PyRandom>() {
        let init = cell.borrow();
        return Ok((init.dtype, InitializerData::RangedRandom { min_len: init.genome_length, max_len: init.genome_length }));
    }
    if initializer.is_callable() {
        // Callable initializer: default to u8 unless the object exposes a dtype attribute
        let dtype = if let Ok(attr) = initializer.getattr("dtype") {
            let s: String = attr.extract()?;
            Dtype::parse(&s)?
        } else {
            Dtype::U8
        };
        return Ok((dtype, InitializerData::PythonCallback(initializer.clone().unbind())));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "initializer must be RangedRandom, Random, or a callable(population_size) -> list[list[...]]",
    ))
}

fn convert_result<T>(
    py: Python<'_>,
    run_result: evolve::collector::standard::RunResult<Vec<T>, f64>,
    fe: &PyFitnessCallback,
    cmp: &PyComparator,
) -> PyResult<PyRunResult>
where
    for<'py> T: IntoPyObject<'py> + Copy,
    for<'py> <T as IntoPyObject<'py>>::Error: std::fmt::Debug,
    PyFitnessCallback: evolve::fitness::FitnessEvaluator<Vec<T>, f64>,
{
    use evolve::fitness::FitnessComparator;

    let population: Vec<PyIndividual> = run_result
        .population()
        .iter()
        .map(|ind| {
            let fitness = *ind.fitness(fe);
            PyIndividual::new(genome_to_pyobject::<T>(py, ind.genome()), fitness)
        })
        .collect();

    let best = run_result
        .population()
        .iter()
        .map(|ind| (ind.genome().clone(), *ind.fitness(fe)))
        .reduce(|a, b| if cmp.is_better(&a.1, &b.1) { a } else { b })
        .map(|(genome, fitness)| PyIndividual::new(genome_to_pyobject::<T>(py, &genome), fitness))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("population is empty"))?;

    Ok(PyRunResult::new(
        population,
        best,
        run_result.generations(),
        run_result.total_duration().as_secs_f64(),
        run_result.best_fitness().to_vec(),
        run_result.generation_durations().iter().map(|d| d.as_secs_f64()).collect(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_ea_inner(
    operators: &Bound<'_, PyAny>,
    fe: PyFitnessCallback,
    termination: PyTermination,
    pop_size: NonZero<usize>,
    rng: SmallRng,
    cmp: PyComparator,
    dtype: Dtype,
    initializer_data: InitializerData,
) -> PyResult<EaInner> {
    macro_rules! make {
        ($extract_fn:ident, $t:ty, $variant:ident) => {{
            let initializer = match initializer_data {
                InitializerData::RangedRandom { min_len, max_len } => {
                    PyInitializer::RangedRandom(RangedRandom::<$t>::new(min_len..max_len + 1))
                }
                InitializerData::PythonCallback(cb) => PyInitializer::PythonCallback(cb),
            };
            Ok(EaInner::$variant(EvolutionaryAlgorithm::new(
                initializer,
                termination,
                fe,
                $extract_fn(operators)?,
                pop_size,
                rng,
                cmp,
            )))
        }};
    }

    match dtype {
        Dtype::U8  => make!(extract_op_u8,  u8,  U8),
        Dtype::U16 => make!(extract_op_u16, u16, U16),
        Dtype::U32 => make!(extract_op_u32, u32, U32),
        Dtype::U64 => make!(extract_op_u64, u64, U64),
        Dtype::I8  => make!(extract_op_i8,  i8,  I8),
        Dtype::I16 => make!(extract_op_i16, i16, I16),
        Dtype::I32 => make!(extract_op_i32, i32, I32),
        Dtype::I64 => make!(extract_op_i64, i64, I64),
        Dtype::F32 => make!(extract_op_f32, f32, F32),
        Dtype::F64 => make!(extract_op_f64, f64, F64),
    }
}
