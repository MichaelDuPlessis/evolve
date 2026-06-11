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

#[pyclass(name = "Tournament")]
pub struct PyTournament {
    tournament_size: usize,
}

#[pymethods]
impl PyTournament {
    #[new]
    fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

#[pyclass(name = "SinglePoint")]
pub struct PySinglePoint;

#[pymethods]
impl PySinglePoint {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "RandomReset")]
pub struct PyRandomReset;

#[pymethods]
impl PyRandomReset {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Stores the Python operator object; Rust tree built lazily in `extract_op`.
#[pyclass(name = "Fill")]
pub struct PyFill {
    operator: PyObject,
}

#[pymethods]
impl PyFill {
    #[new]
    fn new(operator: PyObject) -> Self {
        Self { operator }
    }
}

/// Stores the Python operator object and fixed size; Rust tree built lazily in `extract_op`.
#[pyclass(name = "FillFixed")]
pub struct PyFillFixed {
    operator: PyObject,
    size: usize,
}

#[pymethods]
impl PyFillFixed {
    #[new]
    fn new(operator: PyObject, size: usize) -> Self {
        Self { operator, size }
    }
}

/// Stores the Python list of operators; Rust tree built lazily in `extract_op`.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    operators: Py<PyList>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(operators: Py<PyList>) -> Self {
        Self { operators }
    }
}

/// Stores the Python list of operators; Rust tree built lazily in `extract_op`.
#[pyclass(name = "Combine")]
pub struct PyCombine {
    operators: Py<PyList>,
}

#[pymethods]
impl PyCombine {
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
