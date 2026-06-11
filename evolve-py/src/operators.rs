use std::num::NonZero;

use evolve::{
    core::{context::Context, offspring::Offspring, state::State},
    operators::{
        GeneticOperator,
        sequential::{
            combinator::{
                Combine, Fill, Pipeline,
                fill::{FixedSize, PopSize},
            },
            crossover::SinglePoint,
            mutation::RandomReset,
            selection::Tournament,
        },
    },
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::rngs::SmallRng;

use crate::{comparator::PyComparator, fitness::PyFitnessCallback};

/// Enum that unifies all supported operators for `Vec<u8>` genomes.
pub enum PyOperatorU8 {
    Tournament(Tournament),
    SinglePoint(SinglePoint<u8>),
    RandomReset(RandomReset<u8>),
    Fill(Fill<Box<PyOperatorU8>, PopSize>),
    FillFixed(Fill<Box<PyOperatorU8>, FixedSize>),
    Pipeline(Pipeline<Vec<PyOperatorU8>>),
    Combine(Combine<Vec<PyOperatorU8>>),
}

impl GeneticOperator<Vec<u8>, f64, PyFitnessCallback, SmallRng, PyComparator> for PyOperatorU8 {
    fn apply(
        &self,
        state: &State<Vec<u8>, f64>,
        ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
    ) -> Offspring<Vec<u8>, f64> {
        match self {
            Self::Tournament(op) => op.apply(state, ctx),
            Self::SinglePoint(op) => op.apply(state, ctx),
            Self::RandomReset(op) => op.apply(state, ctx),
            Self::Fill(op) => op.apply(state, ctx),
            Self::FillFixed(op) => op.apply(state, ctx),
            Self::Pipeline(op) => op.apply(state, ctx),
            Self::Combine(op) => op.apply(state, ctx),
        }
    }

    fn transform(
        &self,
        state: State<Vec<u8>, f64>,
        ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
    ) -> Offspring<Vec<u8>, f64> {
        match self {
            Self::Tournament(op) => op.transform(state, ctx),
            Self::SinglePoint(op) => op.transform(state, ctx),
            Self::RandomReset(op) => op.transform(state, ctx),
            Self::Fill(op) => op.transform(state, ctx),
            Self::FillFixed(op) => op.transform(state, ctx),
            Self::Pipeline(op) => op.transform(state, ctx),
            Self::Combine(op) => op.transform(state, ctx),
        }
    }
}

/// Tournament selection — picks the best individual from a random sample of `tournament_size`.
#[pyclass(name = "Tournament")]
pub struct PyTournament {
    tournament_size: usize,
}

#[pymethods]
impl PyTournament {
    /// Create a tournament selector with the given sample size.
    #[new]
    fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

/// Single-point crossover — splits two parents at a random index and swaps the tails.
#[pyclass(name = "SinglePoint")]
pub struct PySinglePoint;

#[pymethods]
impl PySinglePoint {
    /// Create a single-point crossover operator.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Random-reset mutation — replaces a randomly chosen gene with a new random u8 value.
#[pyclass(name = "RandomReset")]
pub struct PyRandomReset;

#[pymethods]
impl PyRandomReset {
    /// Create a random-reset mutation operator.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Repeat the inner operator until the population reaches its target size.
#[pyclass(name = "Fill")]
pub struct PyFill {
    operator: PyObject,
}

#[pymethods]
impl PyFill {
    /// Create a fill operator that wraps `operator` and repeats it to fill the population.
    #[new]
    fn new(operator: PyObject) -> Self {
        Self { operator }
    }
}

/// Repeat the inner operator until exactly `size` individuals are produced.
#[pyclass(name = "FillFixed")]
pub struct PyFillFixed {
    operator: PyObject,
    size: usize,
}

#[pymethods]
impl PyFillFixed {
    /// Create a fill operator that produces exactly `size` individuals using `operator`.
    #[new]
    fn new(operator: PyObject, size: usize) -> Self {
        Self { operator, size }
    }
}

/// Chain operators sequentially: each operator's output is passed as input to the next.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    operators: Py<PyList>,
}

#[pymethods]
impl PyPipeline {
    /// Create a pipeline from a list of operators applied in order.
    #[new]
    fn new(operators: Py<PyList>) -> Self {
        Self { operators }
    }
}

/// Run all operators on the same input population and merge their offspring.
#[pyclass(name = "Combine")]
pub struct PyCombine {
    operators: Py<PyList>,
}

#[pymethods]
impl PyCombine {
    /// Create a combine operator from a list of operators whose outputs are merged.
    #[new]
    fn new(operators: Py<PyList>) -> Self {
        Self { operators }
    }
}

/// Recursively build a `PyOperatorU8` from a Python operator wrapper object.
pub fn extract_op(obj: &Bound<'_, PyAny>) -> PyResult<PyOperatorU8> {
    if let Ok(cell) = obj.downcast::<PyTournament>() {
        let op = cell.borrow();
        return Ok(PyOperatorU8::Tournament(Tournament::new(
            NonZero::new(op.tournament_size).expect("tournament_size must be non-zero"),
        )));
    }
    if obj.downcast::<PySinglePoint>().is_ok() {
        return Ok(PyOperatorU8::SinglePoint(SinglePoint::new()));
    }
    if obj.downcast::<PyRandomReset>().is_ok() {
        return Ok(PyOperatorU8::RandomReset(RandomReset::new()));
    }
    if let Ok(cell) = obj.downcast::<PyPipeline>() {
        let pipe = cell.borrow();
        let py = obj.py();
        let list = pipe.operators.bind(py);
        let ops = extract_ops(list)?;
        return Ok(PyOperatorU8::Pipeline(Pipeline::new(ops)));
    }
    if let Ok(cell) = obj.downcast::<PyCombine>() {
        let comb = cell.borrow();
        let py = obj.py();
        let list = comb.operators.bind(py);
        let ops = extract_ops(list)?;
        return Ok(PyOperatorU8::Combine(Combine::new(ops)));
    }
    if let Ok(cell) = obj.downcast::<PyFill>() {
        let fill = cell.borrow();
        let py = obj.py();
        let inner = extract_op(fill.operator.bind(py))?;
        return Ok(PyOperatorU8::Fill(Fill::from_population_size(Box::new(inner))));
    }
    if let Ok(cell) = obj.downcast::<PyFillFixed>() {
        let fill = cell.borrow();
        let py = obj.py();
        let inner = extract_op(fill.operator.bind(py))?;
        return Ok(PyOperatorU8::FillFixed(Fill::from_fixed_size(Box::new(inner), fill.size)));
    }
    Err(pyo3::exceptions::PyTypeError::new_err("expected an operator"))
}

/// Extract a `Vec<PyOperatorU8>` from a Python list of operator wrappers.
fn extract_ops(list: &Bound<'_, PyList>) -> PyResult<Vec<PyOperatorU8>> {
    list.iter().map(|item| extract_op(&item)).collect()
}
