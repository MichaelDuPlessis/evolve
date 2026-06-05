mod comparator;
mod fitness;

use comparator::{Maximize, Minimize};
use pyo3::prelude::*;

#[pymodule]
fn _evolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Maximize>()?;
    m.add_class::<Minimize>()?;
    Ok(())
}
