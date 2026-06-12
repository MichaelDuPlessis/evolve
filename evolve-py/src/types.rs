use evolve::{
    algorithm::EvolutionaryAlgorithm,
    initialization::RangedRandom,
    termination::MaxGenerations,
};
use rand::rngs::SmallRng;

use crate::{
    comparator::PyComparator,
    fitness::PyFitnessCallback,
    operators::{
        PyOperatorU8, PyOperatorU16, PyOperatorU32, PyOperatorU64,
        PyOperatorI8, PyOperatorI16, PyOperatorI32, PyOperatorI64,
        PyOperatorF32, PyOperatorF64,
    },
};

macro_rules! ea_type {
    ($t:ty, $op:ty) => {
        EvolutionaryAlgorithm<Vec<$t>, f64, RangedRandom<$t>, MaxGenerations, PyFitnessCallback, $op, SmallRng, PyComparator>
    };
}

/// Holds the monomorphized EA for whichever dtype was selected.
/// Both `RangedRandom` and `Random` initializers use `RangedRandom` under the hood.
pub enum EaInner {
    U8(ea_type!(u8,  PyOperatorU8)),
    U16(ea_type!(u16, PyOperatorU16)),
    U32(ea_type!(u32, PyOperatorU32)),
    U64(ea_type!(u64, PyOperatorU64)),
    I8(ea_type!(i8,  PyOperatorI8)),
    I16(ea_type!(i16, PyOperatorI16)),
    I32(ea_type!(i32, PyOperatorI32)),
    I64(ea_type!(i64, PyOperatorI64)),
    F32(ea_type!(f32, PyOperatorF32)),
    F64(ea_type!(f64, PyOperatorF64)),
}

/// Supported genome element dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
}

impl Dtype {
    pub fn parse(s: &str) -> pyo3::PyResult<Self> {
        match s {
            "u8"  => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "i8"  => Ok(Self::I8),
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
