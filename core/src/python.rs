//! Thin Python adapter; listings, path identity and encoding stay in the core.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codec::{self, WireEdge};
use crate::tree::{Listing, Tree};

fn error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn wire(edges: Vec<WireEdge>) -> Vec<(u32, i32)> {
    edges
        .into_iter()
        .map(|edge| (edge.label, edge.odds as i32))
        .collect()
}

#[pyclass(name = "Tree")]
struct PythonTree {
    inner: Tree,
}

#[pymethods]
impl PythonTree {
    #[new]
    fn new(measures: usize) -> Self {
        Self {
            inner: Tree::new(measures),
        }
    }

    fn node(&mut self, path: Vec<u32>) -> PyResult<u32> {
        self.inner.resolve(&path).map_err(error)
    }

    fn path(&self, node: u32) -> PyResult<Vec<u32>> {
        self.inner.path(node).map_err(error)
    }

    fn has_listing(&self, node: u32, measure: usize) -> PyResult<bool> {
        Ok(self.inner.listing(node, measure).map_err(error)?.is_some())
    }

    /// Cache a complete, sorted model distribution without quantizing it first.
    fn set_distribution(
        &mut self,
        node: u32,
        measure: usize,
        labels: Vec<u32>,
        ln_weights: Vec<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_listing(
                node,
                measure,
                Listing {
                    horizon: f64::INFINITY,
                    complete: f64::INFINITY,
                    labels,
                    ln_weights,
                    leftover: 0.0,
                },
            )
            .map_err(error)
    }

    fn clear_listing(&mut self, node: u32, measure: usize) -> PyResult<()> {
        self.inner.clear_listing(node, measure).map_err(error)
    }

    fn expand(&self, node: u32, measure: usize, horizon: u32) -> PyResult<Vec<(u32, i32)>> {
        let listing = self
            .inner
            .listing(node, measure)
            .map_err(error)?
            .ok_or_else(|| PyValueError::new_err("node has no cached distribution"))?;
        if horizon == 0 {
            return Ok(Vec::new());
        }
        Ok(wire(codec::encode(
            &listing.labels,
            &listing.ln_weights,
            f64::from(horizon),
        )))
    }
}

#[pyfunction]
fn encode(labels: Vec<u32>, ln_weights: Vec<f64>, horizon: u32) -> PyResult<Vec<(u32, i32)>> {
    if labels.len() != ln_weights.len() {
        return Err(PyValueError::new_err(
            "labels and log weights must have the same length",
        ));
    }
    Ok(wire(codec::encode(
        &labels,
        &ln_weights,
        f64::from(horizon),
    )))
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    Ok(())
}
