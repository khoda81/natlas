//! Thin Python adapter; listings and node identity stay in the core.
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::tree::{Listing, Tree};

fn error(message: String) -> PyErr {
    PyValueError::new_err(message)
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

    /// The node `path` below the node hashed `hash`. A hash this tree never
    /// learned raises KeyError, which a client answers by naming the node from
    /// further up.
    fn node(&mut self, hash: u64, path: Vec<u32>) -> PyResult<u32> {
        let from = self
            .inner
            .find(hash)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown node {hash:#018x}")))?;
        self.inner.descend(from, &path).map_err(error)
    }

    fn path(&self, node: u32) -> PyResult<Vec<u32>> {
        self.inner.path(node).map_err(error)
    }

    fn has_listing(&self, node: u32, measure: usize) -> PyResult<bool> {
        Ok(self.inner.listing(node, measure).map_err(error)?.is_some())
    }

    /// Cache a complete distribution, heaviest first.
    fn set_distribution(
        &mut self,
        node: u32,
        measure: usize,
        branches: Vec<u32>,
        ln_weights: Vec<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_listing(
                node,
                measure,
                Listing {
                    horizon: f64::INFINITY,
                    complete: f64::INFINITY,
                    branches,
                    ln_weights,
                    leftover: 0.0,
                },
            )
            .map_err(error)
    }

    fn clear_listing(&mut self, node: u32, measure: usize) -> PyResult<()> {
        self.inner.clear_listing(node, measure).map_err(error)
    }

    /// The cached distribution down to 2^-horizon, as (branches, ln weights,
    /// ln weight of the rest) batches of at most `size`.
    #[allow(clippy::type_complexity)]
    fn batches(
        &self,
        node: u32,
        measure: usize,
        horizon: u32,
        size: usize,
    ) -> PyResult<Vec<(Vec<u32>, Vec<f64>, f64)>> {
        let listing = self
            .inner
            .listing(node, measure)
            .map_err(error)?
            .ok_or_else(|| PyValueError::new_err("node has no cached distribution"))?;
        Ok(listing
            .batches(horizon, size)
            .into_iter()
            .map(|(range, ln_remaining)| {
                (
                    listing.branches[range.clone()].to_vec(),
                    listing.ln_weights[range].to_vec(),
                    ln_remaining,
                )
            })
            .collect())
    }
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    Ok(())
}
