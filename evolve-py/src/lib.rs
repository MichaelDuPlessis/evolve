mod algorithm;
mod collector;
mod comparator;
mod experiment;
mod fitness;
mod grammar;
mod initializer;
mod operators;
mod result;
mod termination;
mod types;

use algorithm::PyEvolutionaryAlgorithm;
use comparator::{Maximize, Minimize};
use experiment::PyExperiment;
use grammar::{PyGeFitness, PyGrammar, PyGrammarBuilder, PyStandardMapper};
use initializer::{PyRandom, PyRangedRandom};
use operators::{
    PyArithmetic, PyCombine, PyCreep, PyElitism, PyFill, PyFillFixed, PyGaussian, PyIdentity,
    PyInversion, PyParallelCreep, PyParallelFill, PyParallelGaussian, PyParallelInversion,
    PyParallelRandomReset, PyParallelScramble, PyParallelSwap, PyPipeline, PyProportional,
    PyRandomReset, PyRank, PyRepeat, PyRouletteWheel, PyScramble, PySegmentDeletion,
    PySegmentDuplication, PySinglePoint, PySus, PySwap, PyTournament, PyTwoPoint, PyUniform,
    PyWeighted, PyWithRate,
};
use pyo3::prelude::*;
use result::{PyIndividual, PyRunResult};
use termination::PyMaxGenerations;

#[pymodule]
fn _evolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Maximize>()?;
    m.add_class::<Minimize>()?;
    m.add_class::<PyRangedRandom>()?;
    m.add_class::<PyRandom>()?;
    m.add_class::<PyMaxGenerations>()?;
    // Selection
    m.add_class::<PyTournament>()?;
    m.add_class::<PyElitism>()?;
    m.add_class::<PyRouletteWheel>()?;
    m.add_class::<PyRank>()?;
    m.add_class::<PySus>()?;
    // Crossover
    m.add_class::<PySinglePoint>()?;
    m.add_class::<PyTwoPoint>()?;
    m.add_class::<PyUniform>()?;
    m.add_class::<PyArithmetic>()?;
    // Mutation
    m.add_class::<PyRandomReset>()?;
    m.add_class::<PySwap>()?;
    m.add_class::<PyInversion>()?;
    m.add_class::<PyScramble>()?;
    m.add_class::<PyCreep>()?;
    m.add_class::<PyGaussian>()?;
    m.add_class::<PySegmentDuplication>()?;
    m.add_class::<PySegmentDeletion>()?;
    // Combinators
    m.add_class::<PyFill>()?;
    m.add_class::<PyFillFixed>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyCombine>()?;
    m.add_class::<PyWeighted>()?;
    m.add_class::<PyProportional>()?;
    m.add_class::<PyRepeat>()?;
    m.add_class::<PyIdentity>()?;
    m.add_class::<PyWithRate>()?;
    // Parallel operators
    m.add_class::<PyParallelFill>()?;
    m.add_class::<PyParallelRandomReset>()?;
    m.add_class::<PyParallelSwap>()?;
    m.add_class::<PyParallelInversion>()?;
    m.add_class::<PyParallelScramble>()?;
    m.add_class::<PyParallelCreep>()?;
    m.add_class::<PyParallelGaussian>()?;
    // Grammar / GE
    m.add_class::<PyGrammar>()?;
    m.add_class::<PyGrammarBuilder>()?;
    m.add_class::<PyStandardMapper>()?;
    m.add_class::<PyGeFitness>()?;
    // Core
    m.add_class::<PyEvolutionaryAlgorithm>()?;
    m.add_class::<PyRunResult>()?;
    m.add_class::<PyIndividual>()?;
    m.add_class::<PyExperiment>()?;
    Ok(())
}
