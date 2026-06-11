mod algorithm;
mod comparator;
mod fitness;
mod initializer;
mod operators;
mod result;
mod termination;

use algorithm::PyEvolutionaryAlgorithm;
use comparator::{Maximize, Minimize};
use initializer::PyRangedRandom;
use operators::{PyCombine, PyFill, PyFillFixed, PyPipeline, PyRandomReset, PySinglePoint, PyTournament};
use pyo3::prelude::*;
use result::{PyIndividual, PyRunResult};
use termination::PyMaxGenerations;

#[pymodule]
fn _evolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Maximize>()?;
    m.add_class::<Minimize>()?;
    m.add_class::<PyRangedRandom>()?;
    m.add_class::<PyMaxGenerations>()?;
    m.add_class::<PyTournament>()?;
    m.add_class::<PySinglePoint>()?;
    m.add_class::<PyRandomReset>()?;
    m.add_class::<PyFill>()?;
    m.add_class::<PyFillFixed>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyCombine>()?;
    m.add_class::<PyEvolutionaryAlgorithm>()?;
    m.add_class::<PyRunResult>()?;
    m.add_class::<PyIndividual>()?;
    Ok(())
}
