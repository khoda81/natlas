//! Python adapter for servers: node identity, distributions and Expand's walk
//! stay in the core; a server only computes the distributions a walk asks for.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::distribution::{Distribution, Window};
use crate::tree::{Advance, Tree, Walk};

fn error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass(name = "Tree")]
struct PythonTree {
    inner: Tree,
}

/// One Expand request in progress.
#[pyclass(name = "Walk")]
struct PythonWalk {
    inner: Walk,
}

#[pymethods]
impl PythonTree {
    #[new]
    fn new(measures: usize) -> Self {
        Self {
            inner: Tree::new(measures),
        }
    }

    /// The nodes from `node` down `path`, inclusive. A node this tree never
    /// learned raises KeyError, which a client answers by naming the node
    /// from the root.
    fn walk(&mut self, node: u64, path: Vec<u32>) -> PyResult<Vec<u64>> {
        self.known(node)?;
        let mut nodes = vec![node];
        for branch in path {
            nodes.push(
                self.inner
                    .child(*nodes.last().expect("a start"), branch)
                    .map_err(error)?,
            );
        }
        Ok(nodes)
    }

    fn path(&self, node: u64) -> PyResult<Vec<u32>> {
        self.inner.path(node).map_err(error)
    }

    fn has(&self, node: u64, measure: usize) -> PyResult<bool> {
        Ok(self
            .inner
            .distribution(node, measure)
            .map_err(error)?
            .is_some())
    }

    /// Set the node's children under `measure` from every branch's ln weight,
    /// a float64 buffer indexed by branch, as a measure computes them.
    fn set(
        &mut self,
        node: u64,
        measure: usize,
        ln_weights: PyBuffer<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let weights = ln_weights.to_vec(py)?;
        let distribution = Distribution::from_weights(&weights).map_err(error)?;
        self.inner.set(node, measure, distribution).map_err(error)
    }

    fn forget(&mut self, node: u64, measure: usize) -> PyResult<()> {
        self.inner.forget(node, measure).map_err(error)
    }

    /// Begin an Expand request. An invalid window raises ValueError, an
    /// unknown node KeyError.
    fn start(
        &self,
        node: u64,
        path: Vec<u32>,
        ln_top: f64,
        ln_bottom: f64,
        horizon: u32,
    ) -> PyResult<PythonWalk> {
        self.known(node)?;
        let window = Window::new(ln_top, ln_bottom).map_err(error)?;
        Ok(PythonWalk {
            inner: Walk::new(node, path, window, horizon),
        })
    }

    /// The walk's next message as ("run", node, rank, branches, ln_below),
    /// ("missing", node) when it needs that node's distribution first, or
    /// None once it is done.
    fn advance(
        &mut self,
        walk: &mut PythonWalk,
        measure: usize,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyAny>>> {
        Ok(
            match walk
                .inner
                .advance(&mut self.inner, measure)
                .map_err(error)?
            {
                Advance::Run(node, run) => Some(
                    (
                        "run",
                        node,
                        run.rank(),
                        run.branches().to_vec(),
                        run.ln_below().to_vec(),
                    )
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                ),
                Advance::Missing(node) => {
                    Some(("missing", node).into_pyobject(py)?.into_any().unbind())
                }
                Advance::Done => None,
            },
        )
    }
}

impl PythonTree {
    fn known(&self, node: u64) -> PyResult<()> {
        if self.inner.contains(node) {
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("unknown node {node:#018x}")))
        }
    }
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    module.add_class::<PythonWalk>()?;
    Ok(())
}
