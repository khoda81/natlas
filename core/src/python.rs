//! Python adapter for servers: node identity, distributions and Expand's walk
//! stay in the core; a server only computes the distributions a walk asks for.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::distribution::Distribution;
use crate::tree::{Advance, Read, Tree, Walk};

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

    /// Begin an Expand down `path` below `node`. An unknown node raises
    /// KeyError.
    fn start_path(&self, node: u64, path: Vec<u32>) -> PyResult<PythonWalk> {
        self.start(node, Read::Path(path))
    }

    /// Begin an Expand for the children of `node` holding `points`, each ln
    /// of the mass below it. A point above 0 raises ValueError.
    fn start_points(&self, node: u64, points: Vec<f64>) -> PyResult<PythonWalk> {
        self.start(node, Read::Points(points))
    }

    /// Begin an Expand for the children of `node` along `branches`.
    fn start_branches(&self, node: u64, branches: Vec<u32>) -> PyResult<PythonWalk> {
        self.start(node, Read::Branches(branches))
    }

    /// The walk's next message as ("children", node, runs, children): runs
    /// as (rank, branches, ln_below), children the node's count or None where
    /// its end is unknown; ("missing", node) when it needs that node's
    /// distribution first; or None once it is done.
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
                Advance::Children(node, runs, children) => {
                    let runs: Vec<_> = runs
                        .iter()
                        .map(|run| (run.rank(), run.branches().to_vec(), run.ln_below().to_vec()))
                        .collect();
                    Some(
                        ("children", node, runs, children)
                            .into_pyobject(py)?
                            .into_any()
                            .unbind(),
                    )
                }
                Advance::Missing(node) => {
                    Some(("missing", node).into_pyobject(py)?.into_any().unbind())
                }
                Advance::Done => None,
            },
        )
    }
}

impl PythonTree {
    fn start(&self, node: u64, read: Read) -> PyResult<PythonWalk> {
        self.known(node)?;
        Ok(PythonWalk {
            inner: Walk::new(node, read).map_err(error)?,
        })
    }

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
