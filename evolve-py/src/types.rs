use evolve::{
    algorithm::EvolutionaryAlgorithm,
    core::{context::Context, individual::Individual, population::Population},
    fitness::FitnessEvaluator,
    initialization::{Initializer, RangedRandom},
};
use pyo3::prelude::*;
use rand::{Rng, rngs::SmallRng};
use std::num::NonZero;

use crate::{
    comparator::PyComparator,
    fitness::PyFitnessCallback,
    operators::{
        PyOperatorF32, PyOperatorF64, PyOperatorI8, PyOperatorI16, PyOperatorI32, PyOperatorI64,
        PyOperatorU8, PyOperatorU16, PyOperatorU32, PyOperatorU64,
    },
    termination::PyTermination,
};

macro_rules! ea_type {
    ($t:ty, $op:ty) => {
        EvolutionaryAlgorithm<Vec<$t>, f64, PyInitializer<$t>, PyTermination, PyFitnessCallback, $op, SmallRng, PyComparator>
    };
}

/// Holds the monomorphized EA for whichever dtype was selected.
pub enum EaInner {
    U8(ea_type!(u8, PyOperatorU8)),
    U16(ea_type!(u16, PyOperatorU16)),
    U32(ea_type!(u32, PyOperatorU32)),
    U64(ea_type!(u64, PyOperatorU64)),
    I8(ea_type!(i8, PyOperatorI8)),
    I16(ea_type!(i16, PyOperatorI16)),
    I32(ea_type!(i32, PyOperatorI32)),
    I64(ea_type!(i64, PyOperatorI64)),
    F32(ea_type!(f32, PyOperatorF32)),
    F64(ea_type!(f64, PyOperatorF64)),
}

/// Supported genome element dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Dtype {
    pub fn parse(s: &str) -> pyo3::PyResult<Self> {
        match s {
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid dtype {:?}; expected one of: u8, u16, u32, u64, i8, i16, i32, i64, f32, f64",
                other
            ))),
        }
    }
}

/// A population initializer that wraps either `RangedRandom<T>` or a Python callable.
///
/// The Python callable receives `population_size: int` and must return
/// `list[list[int|float]]` (a list of genomes).
pub enum PyInitializer<T> {
    RangedRandom(RangedRandom<T>),
    PythonCallback(Py<PyAny>),
}

impl<T, F, Fe, R, C> Initializer<Vec<T>, F, Fe, R, C> for PyInitializer<T>
where
    Fe: FitnessEvaluator<Vec<T>, F>,
    T: evolve::random::Randomizable<R> + for<'py> pyo3::FromPyObject<'py>,
    R: Rng,
{
    fn initialize(
        &self,
        population_size: NonZero<usize>,
        ctx: &mut Context<Fe, R, C>,
    ) -> Population<Vec<T>, F> {
        match self {
            Self::RangedRandom(init) => init.initialize(population_size, ctx),
            Self::PythonCallback(cb) => Python::with_gil(|py| {
                let result = match cb.bind(py).call1((population_size.get(),)) {
                    Ok(r) => r,
                    Err(_) => return Population::default(),
                };
                let genomes: Vec<Vec<T>> = match result.extract() {
                    Ok(v) => v,
                    Err(_) => return Population::default(),
                };
                genomes
                    .into_iter()
                    .map(|genome| Individual::new(genome))
                    .collect()
            }),
        }
    }
}
