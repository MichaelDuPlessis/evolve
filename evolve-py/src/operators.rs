use std::num::NonZero;

use evolve::{
    core::{
        context::Context, individual::Individual, offspring::Offspring, population::Population,
        state::State,
    },
    operators::{
        GeneticOperator,
        parallel::{
            combinator::Fill as ParFill,
            mutation::{
                Creep as ParCreep, Gaussian as ParGaussian, Inversion as ParInversion,
                RandomReset as ParRandomReset, Scramble as ParScramble, Swap as ParSwap,
            },
        },
        sequential::{
            combinator::{
                Combine, Fill, Pipeline, Repeat, Weighted,
                fill::{FixedSize, PopSize},
            },
            crossover::{Arithmetic, SinglePoint, TwoPoint, Uniform},
            identity::Identity,
            mutation::{
                Creep, Gaussian, Inversion, RandomReset, Scramble, SegmentDeletion,
                SegmentDuplication, Swap,
            },
            selection::{Elitism, Rank, RouletteWheel, Sus, Tournament},
            with_rate::WithRate,
        },
    },
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::rngs::SmallRng;

use crate::{comparator::PyComparator, fitness::PyFitnessCallback};

// ── Macro: generate PyOperator<Type> enum + GeneticOperator impl ─────────────

/// Generates the integer-dtype operator enum (has Creep, no Gaussian/Arithmetic).
macro_rules! define_int_operator_enum {
    ($name:ident, $t:ty) => {
        pub enum $name {
            Tournament(Tournament),
            SinglePoint(SinglePoint<$t>),
            TwoPoint(TwoPoint<$t>),
            Uniform(Uniform<$t>),
            RandomReset(RandomReset<$t>),
            Swap(Swap<$t>),
            Inversion(Inversion<$t>),
            Scramble(Scramble<$t>),
            Creep(Creep<$t>),
            SegmentDuplication(SegmentDuplication<$t>),
            SegmentDeletion(SegmentDeletion<$t>),
            Elitism(Elitism),
            RouletteWheel(RouletteWheel),
            Rank(Rank),
            Sus(Sus),
            Fill(Fill<Box<$name>, PopSize>),
            FillFixed(Fill<Box<$name>, FixedSize>),
            Pipeline(Pipeline<Vec<$name>>),
            Combine(Combine<Vec<$name>>),
            Weighted(Weighted<Vec<($name, NonZero<u16>)>>),
            Proportional(Box<[($name, NonZero<u16>)]>),
            ProportionalFixed(Box<[($name, NonZero<u16>)]>, usize),
            Repeat(Repeat<Box<$name>>),
            Identity(Identity),
            WithRate(WithRate<Box<$name>>),
            PythonCallback(Py<PyAny>),
            Conditional(Py<PyAny>, Box<$name>, Box<$name>),
            // Parallel operators
            ParallelFill(ParFill<Box<$name>>),
            ParallelRandomReset(ParRandomReset<$t>),
            ParallelSwap(ParSwap<$t>),
            ParallelInversion(ParInversion<$t>),
            ParallelScramble(ParScramble<$t>),
            ParallelCreep(ParCreep<$t>),
        }

        impl GeneticOperator<Vec<$t>, f64, PyFitnessCallback, SmallRng, PyComparator> for $name {
            fn apply(
                &self,
                state: &State<Vec<$t>, f64>,
                ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
            ) -> Offspring<Vec<$t>, f64> {
                match self {
                    Self::Tournament(op) => op.apply(state, ctx),
                    Self::SinglePoint(op) => op.apply(state, ctx),
                    Self::TwoPoint(op) => op.apply(state, ctx),
                    Self::Uniform(op) => op.apply(state, ctx),
                    Self::RandomReset(op) => op.apply(state, ctx),
                    Self::Swap(op) => op.apply(state, ctx),
                    Self::Inversion(op) => op.apply(state, ctx),
                    Self::Scramble(op) => op.apply(state, ctx),
                    Self::Creep(op) => op.apply(state, ctx),
                    Self::SegmentDuplication(op) => op.apply(state, ctx),
                    Self::SegmentDeletion(op) => op.apply(state, ctx),
                    Self::Elitism(op) => op.apply(state, ctx),
                    Self::RouletteWheel(op) => op.apply(state, ctx),
                    Self::Rank(op) => op.apply(state, ctx),
                    Self::Sus(op) => op.apply(state, ctx),
                    Self::Fill(op) => op.apply(state, ctx),
                    Self::FillFixed(op) => op.apply(state, ctx),
                    Self::Pipeline(op) => op.apply(state, ctx),
                    Self::Combine(op) => op.apply(state, ctx),
                    Self::Weighted(op) => op.apply(state, ctx),
                    Self::Proportional(ops) => proportional_apply(ops, state, ctx),
                    Self::ProportionalFixed(ops, size) => proportional_fixed_apply(ops, *size, state, ctx),
                    Self::Repeat(op) => op.apply(state, ctx),
                    Self::Identity(op) => op.apply(state, ctx),
                    Self::WithRate(op) => op.apply(state, ctx),
                    Self::PythonCallback(cb) => python_callback_apply::<$t>(cb, state),
                    Self::Conditional(pred, a, b) => conditional_apply::<$t, $name>(pred, a, b, state, ctx),
                    Self::ParallelFill(op) => op.apply(state, ctx),
                    Self::ParallelRandomReset(op) => op.apply(state, ctx),
                    Self::ParallelSwap(op) => op.apply(state, ctx),
                    Self::ParallelInversion(op) => op.apply(state, ctx),
                    Self::ParallelScramble(op) => op.apply(state, ctx),
                    Self::ParallelCreep(op) => op.apply(state, ctx),
                }
            }

            fn transform(
                &self,
                state: State<Vec<$t>, f64>,
                ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
            ) -> Offspring<Vec<$t>, f64> {
                match self {
                    Self::Tournament(op) => op.transform(state, ctx),
                    Self::SinglePoint(op) => op.transform(state, ctx),
                    Self::TwoPoint(op) => op.transform(state, ctx),
                    Self::Uniform(op) => op.transform(state, ctx),
                    Self::RandomReset(op) => op.transform(state, ctx),
                    Self::Swap(op) => op.transform(state, ctx),
                    Self::Inversion(op) => op.transform(state, ctx),
                    Self::Scramble(op) => op.transform(state, ctx),
                    Self::Creep(op) => op.transform(state, ctx),
                    Self::SegmentDuplication(op) => op.transform(state, ctx),
                    Self::SegmentDeletion(op) => op.transform(state, ctx),
                    Self::Elitism(op) => op.transform(state, ctx),
                    Self::RouletteWheel(op) => op.transform(state, ctx),
                    Self::Rank(op) => op.transform(state, ctx),
                    Self::Sus(op) => op.transform(state, ctx),
                    Self::Fill(op) => op.transform(state, ctx),
                    Self::FillFixed(op) => op.transform(state, ctx),
                    Self::Pipeline(op) => op.transform(state, ctx),
                    Self::Combine(op) => op.transform(state, ctx),
                    Self::Weighted(op) => op.transform(state, ctx),
                    Self::Proportional(ops) => proportional_transform(ops, state, ctx),
                    Self::ProportionalFixed(ops, size) => proportional_fixed_apply(ops, *size, &state, ctx),
                    Self::Repeat(op) => op.transform(state, ctx),
                    Self::Identity(op) => op.transform(state, ctx),
                    Self::WithRate(op) => op.transform(state, ctx),
                    Self::PythonCallback(cb) => python_callback_apply::<$t>(cb, &state),
                    Self::Conditional(pred, a, b) => conditional_apply::<$t, $name>(pred, a, b, &state, ctx),
                    Self::ParallelFill(op) => op.transform(state, ctx),
                    Self::ParallelRandomReset(op) => op.transform(state, ctx),
                    Self::ParallelSwap(op) => op.transform(state, ctx),
                    Self::ParallelInversion(op) => op.transform(state, ctx),
                    Self::ParallelScramble(op) => op.transform(state, ctx),
                    Self::ParallelCreep(op) => op.transform(state, ctx),
                }
            }
        }
    };
}

/// Generates the float-dtype operator enum (has Gaussian + Arithmetic, no Creep).
macro_rules! define_float_operator_enum {
    ($name:ident, $t:ty) => {
        pub enum $name {
            Tournament(Tournament),
            SinglePoint(SinglePoint<$t>),
            TwoPoint(TwoPoint<$t>),
            Uniform(Uniform<$t>),
            Arithmetic(Arithmetic),
            RandomReset(RandomReset<$t>),
            Swap(Swap<$t>),
            Inversion(Inversion<$t>),
            Scramble(Scramble<$t>),
            Gaussian(Gaussian),
            SegmentDuplication(SegmentDuplication<$t>),
            SegmentDeletion(SegmentDeletion<$t>),
            Elitism(Elitism),
            RouletteWheel(RouletteWheel),
            Rank(Rank),
            Sus(Sus),
            Fill(Fill<Box<$name>, PopSize>),
            FillFixed(Fill<Box<$name>, FixedSize>),
            Pipeline(Pipeline<Vec<$name>>),
            Combine(Combine<Vec<$name>>),
            Weighted(Weighted<Vec<($name, NonZero<u16>)>>),
            Proportional(Box<[($name, NonZero<u16>)]>),
            ProportionalFixed(Box<[($name, NonZero<u16>)]>, usize),
            Repeat(Repeat<Box<$name>>),
            Identity(Identity),
            WithRate(WithRate<Box<$name>>),
            PythonCallback(Py<PyAny>),
            Conditional(Py<PyAny>, Box<$name>, Box<$name>),
            // Parallel operators
            ParallelFill(ParFill<Box<$name>>),
            ParallelRandomReset(ParRandomReset<$t>),
            ParallelSwap(ParSwap<$t>),
            ParallelInversion(ParInversion<$t>),
            ParallelScramble(ParScramble<$t>),
            ParallelGaussian(ParGaussian),
        }

        impl GeneticOperator<Vec<$t>, f64, PyFitnessCallback, SmallRng, PyComparator> for $name {
            fn apply(
                &self,
                state: &State<Vec<$t>, f64>,
                ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
            ) -> Offspring<Vec<$t>, f64> {
                match self {
                    Self::Tournament(op) => op.apply(state, ctx),
                    Self::SinglePoint(op) => op.apply(state, ctx),
                    Self::TwoPoint(op) => op.apply(state, ctx),
                    Self::Uniform(op) => op.apply(state, ctx),
                    Self::Arithmetic(op) => op.apply(state, ctx),
                    Self::RandomReset(op) => op.apply(state, ctx),
                    Self::Swap(op) => op.apply(state, ctx),
                    Self::Inversion(op) => op.apply(state, ctx),
                    Self::Scramble(op) => op.apply(state, ctx),
                    Self::Gaussian(op) => op.apply(state, ctx),
                    Self::SegmentDuplication(op) => op.apply(state, ctx),
                    Self::SegmentDeletion(op) => op.apply(state, ctx),
                    Self::Elitism(op) => op.apply(state, ctx),
                    Self::RouletteWheel(op) => op.apply(state, ctx),
                    Self::Rank(op) => op.apply(state, ctx),
                    Self::Sus(op) => op.apply(state, ctx),
                    Self::Fill(op) => op.apply(state, ctx),
                    Self::FillFixed(op) => op.apply(state, ctx),
                    Self::Pipeline(op) => op.apply(state, ctx),
                    Self::Combine(op) => op.apply(state, ctx),
                    Self::Weighted(op) => op.apply(state, ctx),
                    Self::Proportional(ops) => proportional_apply(ops, state, ctx),
                    Self::ProportionalFixed(ops, size) => proportional_fixed_apply(ops, *size, state, ctx),
                    Self::Repeat(op) => op.apply(state, ctx),
                    Self::Identity(op) => op.apply(state, ctx),
                    Self::WithRate(op) => op.apply(state, ctx),
                    Self::PythonCallback(cb) => python_callback_apply::<$t>(cb, state),
                    Self::Conditional(pred, a, b) => conditional_apply::<$t, $name>(pred, a, b, state, ctx),
                    Self::ParallelFill(op) => op.apply(state, ctx),
                    Self::ParallelRandomReset(op) => op.apply(state, ctx),
                    Self::ParallelSwap(op) => op.apply(state, ctx),
                    Self::ParallelInversion(op) => op.apply(state, ctx),
                    Self::ParallelScramble(op) => op.apply(state, ctx),
                    Self::ParallelGaussian(op) => op.apply(state, ctx),
                }
            }

            fn transform(
                &self,
                state: State<Vec<$t>, f64>,
                ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
            ) -> Offspring<Vec<$t>, f64> {
                match self {
                    Self::Tournament(op) => op.transform(state, ctx),
                    Self::SinglePoint(op) => op.transform(state, ctx),
                    Self::TwoPoint(op) => op.transform(state, ctx),
                    Self::Uniform(op) => op.transform(state, ctx),
                    Self::Arithmetic(op) => op.transform(state, ctx),
                    Self::RandomReset(op) => op.transform(state, ctx),
                    Self::Swap(op) => op.transform(state, ctx),
                    Self::Inversion(op) => op.transform(state, ctx),
                    Self::Scramble(op) => op.transform(state, ctx),
                    Self::Gaussian(op) => op.transform(state, ctx),
                    Self::SegmentDuplication(op) => op.transform(state, ctx),
                    Self::SegmentDeletion(op) => op.transform(state, ctx),
                    Self::Elitism(op) => op.transform(state, ctx),
                    Self::RouletteWheel(op) => op.transform(state, ctx),
                    Self::Rank(op) => op.transform(state, ctx),
                    Self::Sus(op) => op.transform(state, ctx),
                    Self::Fill(op) => op.transform(state, ctx),
                    Self::FillFixed(op) => op.transform(state, ctx),
                    Self::Pipeline(op) => op.transform(state, ctx),
                    Self::Combine(op) => op.transform(state, ctx),
                    Self::Weighted(op) => op.transform(state, ctx),
                    Self::Proportional(ops) => proportional_transform(ops, state, ctx),
                    Self::ProportionalFixed(ops, size) => proportional_fixed_apply(ops, *size, &state, ctx),
                    Self::Repeat(op) => op.transform(state, ctx),
                    Self::Identity(op) => op.transform(state, ctx),
                    Self::WithRate(op) => op.transform(state, ctx),
                    Self::PythonCallback(cb) => python_callback_apply::<$t>(cb, &state),
                    Self::Conditional(pred, a, b) => conditional_apply::<$t, $name>(pred, a, b, &state, ctx),
                    Self::ParallelFill(op) => op.transform(state, ctx),
                    Self::ParallelRandomReset(op) => op.transform(state, ctx),
                    Self::ParallelSwap(op) => op.transform(state, ctx),
                    Self::ParallelInversion(op) => op.transform(state, ctx),
                    Self::ParallelScramble(op) => op.transform(state, ctx),
                    Self::ParallelGaussian(op) => op.transform(state, ctx),
                }
            }
        }
    };
}

// Generate all 10 dtype enums
define_int_operator_enum!(PyOperatorU8, u8);
define_int_operator_enum!(PyOperatorU16, u16);
define_int_operator_enum!(PyOperatorU32, u32);
define_int_operator_enum!(PyOperatorU64, u64);
define_int_operator_enum!(PyOperatorI8, i8);
define_int_operator_enum!(PyOperatorI16, i16);
define_int_operator_enum!(PyOperatorI32, i32);
define_int_operator_enum!(PyOperatorI64, i64);
define_float_operator_enum!(PyOperatorF32, f32);
define_float_operator_enum!(PyOperatorF64, f64);

// ── Python callback operator helper ──────────────────────────────────────────

/// Call a Python callable with the population as list of dicts (genome + fitness) and return new offspring.
///
/// The callable receives `list[{"genome": list[T], "fitness": float|None}]` and must return `list[list[T]]`.
fn python_callback_apply<T>(cb: &Py<PyAny>, state: &State<Vec<T>, f64>) -> Offspring<Vec<T>, f64>
where
    T: for<'py> IntoPyObject<'py> + for<'py> pyo3::FromPyObject<'py> + Clone,
    for<'py> <T as IntoPyObject<'py>>::Error: std::fmt::Debug,
{
    use crate::fitness::stash_error;
    use pyo3::BoundObject;
    use pyo3::types::PyDict;
    Python::with_gil(|py| {
        // Build list[{"genome": list[T], "fitness": float|None}]
        let pop_list: Vec<PyObject> = state
            .population()
            .iter()
            .map(|ind| {
                let inner: Vec<PyObject> = ind
                    .genome()
                    .iter()
                    .map(|v| v.clone().into_pyobject(py).unwrap().into_any().unbind())
                    .collect();
                let genome_py = PyList::new(py, inner).unwrap().into_any().unbind();
                let fitness_py: PyObject = match ind.try_fitness() {
                    Some(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
                    None => py.None(),
                };
                let d = PyDict::new(py);
                d.set_item("genome", genome_py).unwrap();
                d.set_item("fitness", fitness_py).unwrap();
                d.into_any().unbind()
            })
            .collect();
        let pop_py = match PyList::new(py, pop_list) {
            Ok(l) => l,
            Err(e) => {
                stash_error(py, e);
                return Offspring::Multiple(Default::default());
            }
        };

        let result = match cb.bind(py).call1((pop_py,)) {
            Ok(r) => r,
            Err(e) => {
                stash_error(py, e);
                return Offspring::Multiple(Default::default());
            }
        };

        let new_genomes: Vec<Vec<T>> = match result.extract() {
            Ok(v) => v,
            Err(_) => {
                stash_error(
                    py,
                    pyo3::exceptions::PyTypeError::new_err(
                        "operator callable must return list[list[...]]",
                    ),
                );
                return Offspring::Multiple(Default::default());
            }
        };

        let population: Population<Vec<T>, f64> = new_genomes
            .into_iter()
            .map(|genome| Individual::new(genome))
            .collect();

        Offspring::Multiple(population)
    })
}

/// Apply a Python-predicated conditional operator.
fn conditional_apply<T, Op>(
    pred: &Py<PyAny>,
    if_true: &Op,
    if_false: &Op,
    state: &State<Vec<T>, f64>,
    ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
) -> Offspring<Vec<T>, f64>
where
    Op: GeneticOperator<Vec<T>, f64, PyFitnessCallback, SmallRng, PyComparator>,
    T: Clone,
{
    use crate::fitness::stash_error;
    let use_true = Python::with_gil(|py| {
        let generation = state.generation();
        let pop_size = state.population().len();
        match pred.bind(py).call1((generation, pop_size)) {
            Ok(r) => r.extract::<bool>().unwrap_or(false),
            Err(e) => { stash_error(py, e); false }
        }
    });
    if use_true { if_true.apply(state, ctx) } else { if_false.apply(state, ctx) }
}

// ── Shared Proportional helpers ───────────────────────────────────────────────

fn proportional_apply<T, Op>(
    ops: &[(Op, NonZero<u16>)],
    state: &State<Vec<T>, f64>,
    ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
) -> Offspring<Vec<T>, f64>
where
    Op: GeneticOperator<Vec<T>, f64, PyFitnessCallback, SmallRng, PyComparator>,
    T: Clone,
{
    let target_size = state.population().len();
    let total_weight: u16 = ops.iter().map(|(_, w)| w.get()).sum();
    let mut population = Population::with_capacity(target_size);
    let mut remaining = target_size;
    for (i, (op, weight)) in ops.iter().enumerate() {
        let target = if i == ops.len() - 1 {
            remaining
        } else {
            let t = (weight.get() as usize * target_size) / total_weight as usize;
            remaining -= t;
            t
        };
        let fill = Fill::from_fixed_size(op, target);
        population.add_offspring(fill.apply(state, ctx));
    }
    Offspring::Multiple(population)
}

fn proportional_transform<T, Op>(
    ops: &[(Op, NonZero<u16>)],
    state: State<Vec<T>, f64>,
    ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
) -> Offspring<Vec<T>, f64>
where
    Op: GeneticOperator<Vec<T>, f64, PyFitnessCallback, SmallRng, PyComparator>,
    T: Clone,
{
    proportional_apply(ops, &state, ctx)
}

fn proportional_fixed_apply<T, Op>(
    ops: &[(Op, NonZero<u16>)],
    target_size: usize,
    state: &State<Vec<T>, f64>,
    ctx: &mut Context<PyFitnessCallback, SmallRng, PyComparator>,
) -> Offspring<Vec<T>, f64>
where
    Op: GeneticOperator<Vec<T>, f64, PyFitnessCallback, SmallRng, PyComparator>,
    T: Clone,
{
    let total_weight: u16 = ops.iter().map(|(_, w)| w.get()).sum();
    let mut population = Population::with_capacity(target_size);
    let mut remaining = target_size;
    for (i, (op, weight)) in ops.iter().enumerate() {
        let target = if i == ops.len() - 1 {
            remaining
        } else {
            let t = (weight.get() as usize * target_size) / total_weight as usize;
            remaining -= t;
            t
        };
        let fill = Fill::from_fixed_size(op, target);
        population.add_offspring(fill.apply(state, ctx));
    }
    Offspring::Multiple(population)
}

// ── Python wrapper structs (dtype-independent) ────────────────────────────────

#[pyclass(name = "Tournament")]
pub struct PyTournament {
    pub tournament_size: usize,
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

#[pyclass(name = "TwoPoint")]
pub struct PyTwoPoint;

#[pymethods]
impl PyTwoPoint {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Uniform")]
pub struct PyUniform;

#[pymethods]
impl PyUniform {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Arithmetic")]
pub struct PyArithmetic;

#[pymethods]
impl PyArithmetic {
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

#[pyclass(name = "Swap")]
pub struct PySwap;

#[pymethods]
impl PySwap {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Inversion")]
pub struct PyInversion;

#[pymethods]
impl PyInversion {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Scramble")]
pub struct PyScramble;

#[pymethods]
impl PyScramble {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Creep")]
pub struct PyCreep {
    /// Step stored as i64 so we can handle all integer dtypes from Python.
    pub step: i64,
}

#[pymethods]
impl PyCreep {
    #[new]
    fn new(step: i64) -> Self {
        Self { step }
    }
}

#[pyclass(name = "Gaussian")]
pub struct PyGaussian {
    pub std_dev: f64,
}

#[pymethods]
impl PyGaussian {
    #[new]
    fn new(std_dev: f64) -> Self {
        Self { std_dev }
    }
}

#[pyclass(name = "SegmentDuplication")]
pub struct PySegmentDuplication {
    pub max_segment_fraction: f64,
    pub max_genome_len: usize,
}

#[pymethods]
impl PySegmentDuplication {
    #[new]
    fn new(max_segment_fraction: f64, max_genome_len: usize) -> Self {
        Self {
            max_segment_fraction,
            max_genome_len,
        }
    }
}

#[pyclass(name = "SegmentDeletion")]
pub struct PySegmentDeletion {
    pub max_segment_fraction: f64,
    pub min_genome_len: usize,
}

#[pymethods]
impl PySegmentDeletion {
    #[new]
    fn new(max_segment_fraction: f64, min_genome_len: usize) -> Self {
        Self {
            max_segment_fraction,
            min_genome_len,
        }
    }
}

#[pyclass(name = "Elitism")]
pub struct PyElitism {
    pub amount: usize,
}

#[pymethods]
impl PyElitism {
    #[new]
    #[pyo3(signature = (amount=1))]
    fn new(amount: usize) -> Self {
        Self { amount }
    }
}

#[pyclass(name = "RouletteWheel")]
pub struct PyRouletteWheel;

#[pymethods]
impl PyRouletteWheel {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Rank")]
pub struct PyRank;

#[pymethods]
impl PyRank {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Sus")]
pub struct PySus {
    pub count: usize,
}

#[pymethods]
impl PySus {
    #[new]
    fn new(count: usize) -> Self {
        Self { count }
    }
}

#[pyclass(name = "Fill")]
pub struct PyFill {
    pub operator: PyObject,
}

#[pymethods]
impl PyFill {
    #[new]
    fn new(operator: PyObject) -> Self {
        Self { operator }
    }
}

#[pyclass(name = "FillFixed")]
pub struct PyFillFixed {
    pub operator: PyObject,
    pub size: usize,
}

#[pymethods]
impl PyFillFixed {
    #[new]
    fn new(operator: PyObject, size: usize) -> Self {
        Self { operator, size }
    }
}

#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    pub operators: Py<PyList>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(operators: Py<PyList>) -> Self {
        Self { operators }
    }
}

#[pyclass(name = "Combine")]
pub struct PyCombine {
    pub operators: Py<PyList>,
}

#[pymethods]
impl PyCombine {
    #[new]
    fn new(operators: Py<PyList>) -> Self {
        Self { operators }
    }
}

#[pyclass(name = "Weighted")]
pub struct PyWeighted {
    pub ops: Vec<(PyObject, u16)>,
}

#[pymethods]
impl PyWeighted {
    #[new]
    fn new(pairs: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut ops = Vec::with_capacity(pairs.len());
        for item in pairs.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let op: PyObject = tuple.get_item(0)?.into();
            let weight: u16 = tuple.get_item(1)?.extract()?;
            ops.push((op, weight));
        }
        Ok(Self { ops })
    }
}

#[pyclass(name = "Proportional")]
pub struct PyProportional {
    pub ops: Vec<(PyObject, u16)>,
    pub size: Option<usize>,
}

#[pymethods]
impl PyProportional {
    #[new]
    #[pyo3(signature = (pairs, size=None))]
    fn new(pairs: &Bound<'_, PyList>, size: Option<usize>) -> PyResult<Self> {
        let mut ops = Vec::with_capacity(pairs.len());
        for item in pairs.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let op: PyObject = tuple.get_item(0)?.into();
            let weight: u16 = tuple.get_item(1)?.extract()?;
            ops.push((op, weight));
        }
        Ok(Self { ops, size })
    }
}

#[pyclass(name = "Conditional")]
pub struct PyConditional {
    pub predicate: PyObject,
    pub if_true: PyObject,
    pub if_false: PyObject,
}

#[pymethods]
impl PyConditional {
    #[new]
    fn new(predicate: PyObject, if_true: PyObject, if_false: PyObject) -> Self {
        Self { predicate, if_true, if_false }
    }
}

#[pyclass(name = "Repeat")]
pub struct PyRepeat {
    pub operator: PyObject,
    pub n: usize,
}

#[pymethods]
impl PyRepeat {
    #[new]
    fn new(operator: PyObject, n: usize) -> Self {
        Self { operator, n }
    }
}

#[pyclass(name = "Identity")]
pub struct PyIdentity;

#[pymethods]
impl PyIdentity {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "WithRate")]
pub struct PyWithRate {
    pub operator: PyObject,
    pub rate: f64,
}

#[pymethods]
impl PyWithRate {
    #[new]
    fn new(operator: PyObject, rate: f64) -> Self {
        Self { operator, rate }
    }
}

// ── Parallel operator Python wrappers ────────────────────────────────────────

/// Parallel version of Fill. Distributes offspring generation across threads.
#[pyclass(name = "ParallelFill")]
pub struct PyParallelFill {
    pub operator: PyObject,
    pub target_size: usize,
}

#[pymethods]
impl PyParallelFill {
    #[new]
    fn new(operator: PyObject, target_size: usize) -> Self {
        Self {
            operator,
            target_size,
        }
    }
}

/// Parallel version of RandomReset mutation.
#[pyclass(name = "ParallelRandomReset")]
pub struct PyParallelRandomReset;

#[pymethods]
impl PyParallelRandomReset {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Parallel version of Swap mutation.
#[pyclass(name = "ParallelSwap")]
pub struct PyParallelSwap;

#[pymethods]
impl PyParallelSwap {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Parallel version of Inversion mutation.
#[pyclass(name = "ParallelInversion")]
pub struct PyParallelInversion;

#[pymethods]
impl PyParallelInversion {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Parallel version of Scramble mutation.
#[pyclass(name = "ParallelScramble")]
pub struct PyParallelScramble;

#[pymethods]
impl PyParallelScramble {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Parallel version of Creep mutation (integer dtypes only).
#[pyclass(name = "ParallelCreep")]
pub struct PyParallelCreep {
    pub step: i64,
}

#[pymethods]
impl PyParallelCreep {
    #[new]
    fn new(step: i64) -> Self {
        Self { step }
    }
}

/// Parallel version of Gaussian mutation (float dtypes only).
#[pyclass(name = "ParallelGaussian")]
pub struct PyParallelGaussian {
    pub std_dev: f64,
}

#[pymethods]
impl PyParallelGaussian {
    #[new]
    fn new(std_dev: f64) -> Self {
        Self { std_dev }
    }
}

// ── extract_op_* functions (one per dtype) ────────────────────────────────────

/// Macro that generates an `extract_op_*` function for an integer dtype.
macro_rules! extract_int_op {
    ($fn_name:ident, $enum_name:ident, $t:ty, $creep_cast:expr) => {
        pub fn $fn_name(obj: &Bound<'_, PyAny>) -> PyResult<$enum_name> {
            // Selection
            if let Ok(cell) = obj.downcast::<PyTournament>() {
                let op = cell.borrow();
                return Ok($enum_name::Tournament(Tournament::new(
                    NonZero::new(op.tournament_size).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("tournament_size must be non-zero")
                    })?,
                )));
            }
            if let Ok(cell) = obj.downcast::<PyElitism>() {
                let op = cell.borrow();
                return Ok($enum_name::Elitism(Elitism::new(
                    NonZero::new(op.amount).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("amount must be non-zero")
                    })?,
                )));
            }
            if obj.downcast::<PyRouletteWheel>().is_ok() {
                return Ok($enum_name::RouletteWheel(RouletteWheel));
            }
            if obj.downcast::<PyRank>().is_ok() {
                return Ok($enum_name::Rank(Rank));
            }
            if let Ok(cell) = obj.downcast::<PySus>() {
                let op = cell.borrow();
                return Ok($enum_name::Sus(Sus::new(
                    NonZero::new(op.count).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("count must be non-zero")
                    })?,
                )));
            }
            // Crossover
            if obj.downcast::<PySinglePoint>().is_ok() {
                return Ok($enum_name::SinglePoint(SinglePoint::new()));
            }
            if obj.downcast::<PyTwoPoint>().is_ok() {
                return Ok($enum_name::TwoPoint(TwoPoint::<$t>::new()));
            }
            if obj.downcast::<PyUniform>().is_ok() {
                return Ok($enum_name::Uniform(Uniform::<$t>::new()));
            }
            if obj.downcast::<PyArithmetic>().is_ok() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Arithmetic crossover is only supported for f32/f64 dtypes",
                ));
            }
            // Mutation
            if obj.downcast::<PyRandomReset>().is_ok() {
                return Ok($enum_name::RandomReset(RandomReset::new()));
            }
            if obj.downcast::<PySwap>().is_ok() {
                return Ok($enum_name::Swap(Swap::<$t>::new()));
            }
            if obj.downcast::<PyInversion>().is_ok() {
                return Ok($enum_name::Inversion(Inversion::<$t>::new()));
            }
            if obj.downcast::<PyScramble>().is_ok() {
                return Ok($enum_name::Scramble(Scramble::<$t>::new()));
            }
            if let Ok(cell) = obj.downcast::<PyCreep>() {
                let op = cell.borrow();
                return Ok($enum_name::Creep(Creep::<$t>::new($creep_cast(op.step))));
            }
            if obj.downcast::<PyGaussian>().is_ok() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Gaussian mutation is only supported for f32/f64 dtypes",
                ));
            }
            if let Ok(cell) = obj.downcast::<PySegmentDuplication>() {
                let op = cell.borrow();
                return Ok($enum_name::SegmentDuplication(
                    SegmentDuplication::<$t>::new(op.max_segment_fraction, op.max_genome_len),
                ));
            }
            if let Ok(cell) = obj.downcast::<PySegmentDeletion>() {
                let op = cell.borrow();
                return Ok($enum_name::SegmentDeletion(SegmentDeletion::<$t>::new(
                    op.max_segment_fraction,
                    op.min_genome_len,
                )));
            }
            // Combinators
            if let Ok(cell) = obj.downcast::<PyFill>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::Fill(Fill::from_population_size(Box::new(
                    inner,
                ))));
            }
            if let Ok(cell) = obj.downcast::<PyFillFixed>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::FillFixed(Fill::from_fixed_size(
                    Box::new(inner),
                    fill.size,
                )));
            }
            if let Ok(cell) = obj.downcast::<PyPipeline>() {
                let pipe = cell.borrow();
                let list = pipe.operators.bind(obj.py());
                let ops: PyResult<Vec<$enum_name>> =
                    list.iter().map(|item| $fn_name(&item)).collect();
                return Ok($enum_name::Pipeline(Pipeline::new(ops?)));
            }
            if let Ok(cell) = obj.downcast::<PyCombine>() {
                let comb = cell.borrow();
                let list = comb.operators.bind(obj.py());
                let ops: PyResult<Vec<$enum_name>> =
                    list.iter().map(|item| $fn_name(&item)).collect();
                return Ok($enum_name::Combine(Combine::new(ops?)));
            }
            if let Ok(cell) = obj.downcast::<PyWeighted>() {
                let w = cell.borrow();
                let pairs: PyResult<Vec<($enum_name, NonZero<u16>)>> = w
                    .ops
                    .iter()
                    .map(|(op_obj, weight)| {
                        let inner = $fn_name(op_obj.bind(obj.py()))?;
                        let nz = NonZero::new(*weight).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("weight must be non-zero")
                        })?;
                        Ok((inner, nz))
                    })
                    .collect();
                return Ok($enum_name::Weighted(Weighted::new(pairs?)));
            }
            if let Ok(cell) = obj.downcast::<PyProportional>() {
                let p = cell.borrow();
                let pairs: PyResult<Vec<($enum_name, NonZero<u16>)>> = p
                    .ops
                    .iter()
                    .map(|(op_obj, weight)| {
                        let inner = $fn_name(op_obj.bind(obj.py()))?;
                        let nz = NonZero::new(*weight).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("weight must be non-zero")
                        })?;
                        Ok((inner, nz))
                    })
                    .collect();
                return match p.size {
                    None => Ok($enum_name::Proportional(pairs?.into_boxed_slice())),
                    Some(size) => Ok($enum_name::ProportionalFixed(pairs?.into_boxed_slice(), size)),
                };
            }
            if let Ok(cell) = obj.downcast::<PyConditional>() {
                let c = cell.borrow();
                let a = $fn_name(c.if_true.bind(obj.py()))?;
                let b = $fn_name(c.if_false.bind(obj.py()))?;
                return Ok($enum_name::Conditional(c.predicate.clone_ref(obj.py()), Box::new(a), Box::new(b)));
            }
            if let Ok(cell) = obj.downcast::<PyRepeat>() {
                let r = cell.borrow();
                let inner = $fn_name(r.operator.bind(obj.py()))?;
                return Ok($enum_name::Repeat(Repeat::new(Box::new(inner), r.n)));
            }
            if obj.downcast::<PyIdentity>().is_ok() {
                return Ok($enum_name::Identity(Identity::new()));
            }
            if let Ok(cell) = obj.downcast::<PyWithRate>() {
                let wr = cell.borrow();
                let inner = $fn_name(wr.operator.bind(obj.py()))?;
                return Ok($enum_name::WithRate(WithRate::new(
                    Box::new(inner),
                    wr.rate,
                )));
            }
            // Parallel operators
            if let Ok(cell) = obj.downcast::<PyParallelFill>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::ParallelFill(ParFill::new(
                    Box::new(inner),
                    fill.target_size,
                )));
            }
            if obj.downcast::<PyParallelRandomReset>().is_ok() {
                return Ok($enum_name::ParallelRandomReset(ParRandomReset::new()));
            }
            if obj.downcast::<PyParallelSwap>().is_ok() {
                return Ok($enum_name::ParallelSwap(ParSwap::new()));
            }
            if obj.downcast::<PyParallelInversion>().is_ok() {
                return Ok($enum_name::ParallelInversion(ParInversion::new()));
            }
            if obj.downcast::<PyParallelScramble>().is_ok() {
                return Ok($enum_name::ParallelScramble(ParScramble::new()));
            }
            if let Ok(cell) = obj.downcast::<PyParallelCreep>() {
                let op = cell.borrow();
                return Ok($enum_name::ParallelCreep(ParCreep::new($creep_cast(
                    op.step,
                ))));
            }
            if obj.downcast::<PyParallelGaussian>().is_ok() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ParallelGaussian is only supported for f32/f64 dtypes",
                ));
            }
            // Python callable fallback
            if obj.is_callable() {
                return Ok($enum_name::PythonCallback(obj.clone().unbind()));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected an operator or callable",
            ))
        }
    };
}

/// Macro that generates an `extract_op_*` function for a float dtype.
macro_rules! extract_float_op {
    ($fn_name:ident, $enum_name:ident, $t:ty) => {
        pub fn $fn_name(obj: &Bound<'_, PyAny>) -> PyResult<$enum_name> {
            // Selection
            if let Ok(cell) = obj.downcast::<PyTournament>() {
                let op = cell.borrow();
                return Ok($enum_name::Tournament(Tournament::new(
                    NonZero::new(op.tournament_size).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("tournament_size must be non-zero")
                    })?,
                )));
            }
            if let Ok(cell) = obj.downcast::<PyElitism>() {
                let op = cell.borrow();
                return Ok($enum_name::Elitism(Elitism::new(
                    NonZero::new(op.amount).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("amount must be non-zero")
                    })?,
                )));
            }
            if obj.downcast::<PyRouletteWheel>().is_ok() {
                return Ok($enum_name::RouletteWheel(RouletteWheel));
            }
            if obj.downcast::<PyRank>().is_ok() {
                return Ok($enum_name::Rank(Rank));
            }
            if let Ok(cell) = obj.downcast::<PySus>() {
                let op = cell.borrow();
                return Ok($enum_name::Sus(Sus::new(
                    NonZero::new(op.count).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("count must be non-zero")
                    })?,
                )));
            }
            // Crossover
            if obj.downcast::<PySinglePoint>().is_ok() {
                return Ok($enum_name::SinglePoint(SinglePoint::new()));
            }
            if obj.downcast::<PyTwoPoint>().is_ok() {
                return Ok($enum_name::TwoPoint(TwoPoint::<$t>::new()));
            }
            if obj.downcast::<PyUniform>().is_ok() {
                return Ok($enum_name::Uniform(Uniform::<$t>::new()));
            }
            if obj.downcast::<PyArithmetic>().is_ok() {
                return Ok($enum_name::Arithmetic(Arithmetic::new()));
            }
            // Mutation
            if obj.downcast::<PyRandomReset>().is_ok() {
                return Ok($enum_name::RandomReset(RandomReset::new()));
            }
            if obj.downcast::<PySwap>().is_ok() {
                return Ok($enum_name::Swap(Swap::<$t>::new()));
            }
            if obj.downcast::<PyInversion>().is_ok() {
                return Ok($enum_name::Inversion(Inversion::<$t>::new()));
            }
            if obj.downcast::<PyScramble>().is_ok() {
                return Ok($enum_name::Scramble(Scramble::<$t>::new()));
            }
            if obj.downcast::<PyCreep>().is_ok() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Creep mutation is only supported for integer dtypes",
                ));
            }
            if let Ok(cell) = obj.downcast::<PyGaussian>() {
                let op = cell.borrow();
                return Ok($enum_name::Gaussian(Gaussian::new(op.std_dev)));
            }
            if let Ok(cell) = obj.downcast::<PySegmentDuplication>() {
                let op = cell.borrow();
                return Ok($enum_name::SegmentDuplication(
                    SegmentDuplication::<$t>::new(op.max_segment_fraction, op.max_genome_len),
                ));
            }
            if let Ok(cell) = obj.downcast::<PySegmentDeletion>() {
                let op = cell.borrow();
                return Ok($enum_name::SegmentDeletion(SegmentDeletion::<$t>::new(
                    op.max_segment_fraction,
                    op.min_genome_len,
                )));
            }
            // Combinators
            if let Ok(cell) = obj.downcast::<PyFill>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::Fill(Fill::from_population_size(Box::new(
                    inner,
                ))));
            }
            if let Ok(cell) = obj.downcast::<PyFillFixed>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::FillFixed(Fill::from_fixed_size(
                    Box::new(inner),
                    fill.size,
                )));
            }
            if let Ok(cell) = obj.downcast::<PyPipeline>() {
                let pipe = cell.borrow();
                let list = pipe.operators.bind(obj.py());
                let ops: PyResult<Vec<$enum_name>> =
                    list.iter().map(|item| $fn_name(&item)).collect();
                return Ok($enum_name::Pipeline(Pipeline::new(ops?)));
            }
            if let Ok(cell) = obj.downcast::<PyCombine>() {
                let comb = cell.borrow();
                let list = comb.operators.bind(obj.py());
                let ops: PyResult<Vec<$enum_name>> =
                    list.iter().map(|item| $fn_name(&item)).collect();
                return Ok($enum_name::Combine(Combine::new(ops?)));
            }
            if let Ok(cell) = obj.downcast::<PyWeighted>() {
                let w = cell.borrow();
                let pairs: PyResult<Vec<($enum_name, NonZero<u16>)>> = w
                    .ops
                    .iter()
                    .map(|(op_obj, weight)| {
                        let inner = $fn_name(op_obj.bind(obj.py()))?;
                        let nz = NonZero::new(*weight).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("weight must be non-zero")
                        })?;
                        Ok((inner, nz))
                    })
                    .collect();
                return Ok($enum_name::Weighted(Weighted::new(pairs?)));
            }
            if let Ok(cell) = obj.downcast::<PyProportional>() {
                let p = cell.borrow();
                let pairs: PyResult<Vec<($enum_name, NonZero<u16>)>> = p
                    .ops
                    .iter()
                    .map(|(op_obj, weight)| {
                        let inner = $fn_name(op_obj.bind(obj.py()))?;
                        let nz = NonZero::new(*weight).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("weight must be non-zero")
                        })?;
                        Ok((inner, nz))
                    })
                    .collect();
                return match p.size {
                    None => Ok($enum_name::Proportional(pairs?.into_boxed_slice())),
                    Some(size) => Ok($enum_name::ProportionalFixed(pairs?.into_boxed_slice(), size)),
                };
            }
            if let Ok(cell) = obj.downcast::<PyConditional>() {
                let c = cell.borrow();
                let a = $fn_name(c.if_true.bind(obj.py()))?;
                let b = $fn_name(c.if_false.bind(obj.py()))?;
                return Ok($enum_name::Conditional(c.predicate.clone_ref(obj.py()), Box::new(a), Box::new(b)));
            }
            if let Ok(cell) = obj.downcast::<PyRepeat>() {
                let r = cell.borrow();
                let inner = $fn_name(r.operator.bind(obj.py()))?;
                return Ok($enum_name::Repeat(Repeat::new(Box::new(inner), r.n)));
            }
            if obj.downcast::<PyIdentity>().is_ok() {
                return Ok($enum_name::Identity(Identity::new()));
            }
            if let Ok(cell) = obj.downcast::<PyWithRate>() {
                let wr = cell.borrow();
                let inner = $fn_name(wr.operator.bind(obj.py()))?;
                return Ok($enum_name::WithRate(WithRate::new(
                    Box::new(inner),
                    wr.rate,
                )));
            }
            // Parallel operators
            if let Ok(cell) = obj.downcast::<PyParallelFill>() {
                let fill = cell.borrow();
                let inner = $fn_name(fill.operator.bind(obj.py()))?;
                return Ok($enum_name::ParallelFill(ParFill::new(
                    Box::new(inner),
                    fill.target_size,
                )));
            }
            if obj.downcast::<PyParallelRandomReset>().is_ok() {
                return Ok($enum_name::ParallelRandomReset(ParRandomReset::new()));
            }
            if obj.downcast::<PyParallelSwap>().is_ok() {
                return Ok($enum_name::ParallelSwap(ParSwap::new()));
            }
            if obj.downcast::<PyParallelInversion>().is_ok() {
                return Ok($enum_name::ParallelInversion(ParInversion::new()));
            }
            if obj.downcast::<PyParallelScramble>().is_ok() {
                return Ok($enum_name::ParallelScramble(ParScramble::new()));
            }
            if obj.downcast::<PyParallelCreep>().is_ok() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ParallelCreep is only supported for integer dtypes",
                ));
            }
            if let Ok(cell) = obj.downcast::<PyParallelGaussian>() {
                let op = cell.borrow();
                return Ok($enum_name::ParallelGaussian(ParGaussian::new(op.std_dev)));
            }
            // Python callable fallback
            if obj.is_callable() {
                return Ok($enum_name::PythonCallback(obj.clone().unbind()));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected an operator or callable",
            ))
        }
    };
}

// Generate extract functions for all 10 dtypes
extract_int_op!(extract_op_u8, PyOperatorU8, u8, |s: i64| s as u8);
extract_int_op!(extract_op_u16, PyOperatorU16, u16, |s: i64| s as u16);
extract_int_op!(extract_op_u32, PyOperatorU32, u32, |s: i64| s as u32);
extract_int_op!(extract_op_u64, PyOperatorU64, u64, |s: i64| s as u64);
extract_int_op!(extract_op_i8, PyOperatorI8, i8, |s: i64| s as i8);
extract_int_op!(extract_op_i16, PyOperatorI16, i16, |s: i64| s as i16);
extract_int_op!(extract_op_i32, PyOperatorI32, i32, |s: i64| s as i32);
extract_int_op!(extract_op_i64, PyOperatorI64, i64, |s: i64| s);
extract_float_op!(extract_op_f32, PyOperatorF32, f32);
extract_float_op!(extract_op_f64, PyOperatorF64, f64);
