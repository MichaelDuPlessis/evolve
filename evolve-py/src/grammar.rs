//! Python wrappers for Grammar, StandardMapper, and GeFitness.

use evolve::grammar::{Grammar, GrammarBuilder};
use evolve::grammar::mapper::{Mapper, StandardMapper};
use evolve::phenotype::{Event, PhenotypeBuilder};
use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};

// --- StringBuilder ---

/// Simple phenotype builder that concatenates terminal strings.
#[derive(Default)]
struct StringBuilder(String);

impl PhenotypeBuilder<String> for StringBuilder {
    type Output = String;

    fn push(&mut self, event: Event<String>) {
        if let Event::Terminal(t) = event {
            self.0.push_str(&t);
        }
    }

    fn finish(self) -> String {
        self.0
    }
}

// --- PyGrammar ---

/// A context-free grammar for grammatical evolution.
///
/// Build with `Grammar.builder()`.
#[pyclass(name = "Grammar")]
pub struct PyGrammar {
    pub inner: Grammar<String>,
}

#[pymethods]
impl PyGrammar {
    /// Returns a new GrammarBuilder.
    #[staticmethod]
    fn builder() -> PyGrammarBuilder {
        PyGrammarBuilder {
            inner: Some(GrammarBuilder::new()),
        }
    }
}

// --- PyGrammarBuilder ---

/// Fluent builder for Grammar.
///
/// Each method returns a new GrammarBuilder (consuming the previous one).
#[pyclass(name = "GrammarBuilder")]
pub struct PyGrammarBuilder {
    /// Option so we can take ownership when calling consuming builder methods.
    inner: Option<GrammarBuilder<String>>,
}

impl PyGrammarBuilder {
    fn take(&mut self) -> PyResult<GrammarBuilder<String>> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("GrammarBuilder has already been consumed (called build() or reused)")
        })
    }
}

#[pymethods]
impl PyGrammarBuilder {
    /// Add a rule: `.rule("name", [["sym", ...], ...])`
    fn rule(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        productions: Vec<Vec<String>>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let builder = slf.take()?;
        let refs: Vec<&[String]> = productions.iter().map(|p| p.as_slice()).collect();
        let new_builder = builder.rule(name, &refs);
        Ok(Bound::new(py, PyGrammarBuilder { inner: Some(new_builder) })?.into_any().unbind())
    }

    /// Set the start symbol.
    fn start(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let builder = slf.take()?;
        let new_builder = builder.start(name);
        Ok(Bound::new(py, PyGrammarBuilder { inner: Some(new_builder) })?.into_any().unbind())
    }

    /// Build the Grammar.
    fn build(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let builder = slf.take()?;
        let result = std::panic::catch_unwind(|| builder.build());
        match result {
            Ok(grammar) => Ok(Bound::new(py, PyGrammar { inner: grammar })?.into_any().unbind()),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("grammar build failed");
                Err(PyValueError::new_err(msg.to_string()))
            }
        }
    }
}

// --- PyStandardMapper ---

/// The standard GE codon mapping algorithm.
#[pyclass(name = "StandardMapper")]
pub struct PyStandardMapper {
    pub inner: StandardMapper,
}

#[pymethods]
impl PyStandardMapper {
    #[new]
    #[pyo3(signature = (max_wraps=3))]
    fn new(max_wraps: usize) -> Self {
        Self {
            inner: StandardMapper::new(max_wraps),
        }
    }
}

// --- PyGeFitness ---

enum MapperKind {
    Standard(StandardMapper),
    Python(Py<PyAny>),
}

/// GE fitness: grammar + mapper + Python evaluator + penalty.
///
/// Implements `__call__(genome: list[int]) -> float` so it can be passed
/// directly as the `fitness` argument to `EvolutionaryAlgorithm`.
#[pyclass(name = "GeFitness")]
pub struct PyGeFitness {
    grammar: Grammar<String>,
    mapper: MapperKind,
    evaluator: Py<PyAny>,
    penalty: f64,
}

#[pymethods]
impl PyGeFitness {
    #[new]
    #[pyo3(signature = (grammar, mapper, evaluator, penalty=0.0))]
    fn new(
        grammar: &PyGrammar,
        mapper: &Bound<'_, PyAny>,
        evaluator: Py<PyAny>,
        penalty: f64,
    ) -> PyResult<Self> {
        let mapper_kind = if let Ok(m) = mapper.downcast::<PyStandardMapper>() {
            MapperKind::Standard(m.borrow().inner)
        } else if mapper.is_callable() {
            MapperKind::Python(mapper.clone().unbind())
        } else {
            return Err(PyTypeError::new_err(
                "mapper must be StandardMapper or a callable",
            ));
        };

        Ok(Self {
            grammar: grammar.inner.clone(),
            mapper: mapper_kind,
            evaluator,
            penalty,
        })
    }

    /// Evaluate a genome (list of ints) — makes this callable as a fitness function.
    fn __call__(&self, genome: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<f64> {
        let codons: Vec<u8> = genome.extract()?;
        Ok(self.evaluate_u8(&codons, py))
    }
}

impl PyGeFitness {
    /// Evaluate a u8 codon sequence against the grammar.
    pub fn evaluate_u8(&self, codons: &[u8], py: Python<'_>) -> f64 {
        self.map_and_evaluate(codons, py)
    }

    fn map_and_evaluate(&self, codons: &[u8], py: Python<'_>) -> f64 {
        match &self.mapper {
            MapperKind::Standard(m) => {
                match m.map(&self.grammar, codons, StringBuilder::default()) {
                    Some(phenotype) => self.call_evaluator(py, &phenotype),
                    None => self.penalty,
                }
            }
            MapperKind::Python(mapper_fn) => {
                let py_codons = pyo3::types::PyList::new(
                    py,
                    codons.iter().map(|&v| v.into_pyobject(py).unwrap()),
                ).unwrap();
                match mapper_fn.bind(py).call1((py_codons,)) {
                    Ok(r) if !r.is_none() => {
                        match r.extract::<String>() {
                            Ok(phenotype) => self.call_evaluator(py, &phenotype),
                            Err(_) => self.penalty,
                        }
                    }
                    _ => self.penalty,
                }
            }
        }
    }

    fn call_evaluator(&self, py: Python<'_>, phenotype: &str) -> f64 {
        self.evaluator
            .bind(py)
            .call1((phenotype,))
            .expect("GE evaluator raised an exception")
            .extract::<f64>()
            .expect("GE evaluator must return a float")
    }
}
