use pyo3::prelude::*;
use std::collections::HashMap;

mod bpe;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn train_bpe(
    input_path: String,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
    bpe::train_bpe(&input_path, vocab_size, &special_tokens)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pymodule]
fn cs336_basics_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}
