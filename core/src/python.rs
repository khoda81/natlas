//! Thin Python adapter: node identity stays in the core. A server keeps its
//! distributions on its own device and cuts them there.
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::tree::Tree;

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
    fn new() -> Self {
        Self {
            inner: Tree::new(0),
        }
    }

    /// The nodes from the node hashed `hash` down `path`, inclusive. A hash
    /// this tree never learned raises KeyError, which a client answers by
    /// naming the node from the root.
    fn walk(&mut self, hash: u64, path: Vec<u32>) -> PyResult<Vec<u32>> {
        let mut node = self
            .inner
            .find(hash)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown node {hash:#018x}")))?;
        let mut nodes = Vec::with_capacity(path.len() + 1);
        nodes.push(node);
        for branch in path {
            node = self.inner.child(node, branch).map_err(error)?;
            nodes.push(node);
        }
        Ok(nodes)
    }

    fn path(&self, node: u32) -> PyResult<Vec<u32>> {
        self.inner.path(node).map_err(error)
    }
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    Ok(())
}
