//! Python adapter for servers: node identity, distributions and what each
//! client's view lacks stay in the core; a server only computes the
//! distributions its sessions wait on.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::distribution::{Distribution, Run};
use crate::session::{Demand, Path, Session, Update, View};
use crate::tree::{Lens, Tree};

fn error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass(name = "Tree")]
struct PythonTree {
    inner: Tree,
}

/// One client's session: what it is looking at, read through a tree, and
/// what it has been sent.
#[pyclass(name = "Session")]
struct PythonSession {
    inner: Session,
}

#[pymethods]
impl PythonTree {
    #[new]
    fn new() -> Self {
        Self { inner: Tree::new() }
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

    /// Set each node's ordered branch and final log-weight pairs, packed
    /// into parallel u32 and f64 buffers. Packing runs without the GIL.
    fn set(
        slf: &Bound<'_, Self>,
        nodes: Vec<u64>,
        measure: usize,
        children: Vec<(PyBuffer<u32>, PyBuffer<f64>)>,
    ) -> PyResult<Vec<usize>> {
        if nodes.len() != children.len() {
            return Err(error(format!(
                "{} nodes but {} rows of children",
                nodes.len(),
                children.len()
            )));
        }
        let py = slf.py();
        let rows = children
            .into_iter()
            .map(|(branches, weights)| {
                if branches.item_count() != weights.item_count() {
                    return Err(error("each branch needs one log weight".into()));
                }
                Ok(branches
                    .to_vec(py)?
                    .into_iter()
                    .zip(weights.to_vec(py)?)
                    .collect())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let distributions = py.detach(|| packed(rows)).map_err(error)?;
        let mut tree = slf.borrow_mut();
        nodes
            .into_iter()
            .zip(distributions)
            .map(|(node, distribution)| {
                let bytes = distribution.bytes();
                tree.inner.set(node, measure, distribution).map_err(error)?;
                Ok(bytes)
            })
            .collect()
    }

    fn forget(&mut self, node: u64, measure: usize) -> PyResult<()> {
        self.inner.forget(node, measure).map_err(error)
    }
}

/// Each row's distribution, rows shared out among the cores in order.
fn packed(mut rows: Vec<Vec<(u32, f64)>>) -> Result<Vec<Distribution>, String> {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let share = rows.len().div_ceil(cores).max(1);
    let count = rows.len();
    std::thread::scope(|scope| {
        let workers: Vec<_> = rows
            .chunks_mut(share)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter_mut()
                        .map(|row| Distribution::from_ordered(std::mem::take(row)))
                        .collect::<Result<Vec<_>, String>>()
                })
            })
            .collect();
        let mut distributions = Vec::with_capacity(count);
        for worker in workers {
            distributions.extend(worker.join().expect("packing does not panic")?);
        }
        Ok(distributions)
    })
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

/// A run as (rank, branches, ln_weights, top digits).
fn run(run: &Run) -> (u32, Vec<u32>, Vec<f64>, Vec<u8>) {
    (
        run.rank(),
        run.branches().to_vec(),
        run.ln_weights().to_vec(),
        run.top_digits(),
    )
}

#[pymethods]
impl PythonSession {
    #[new]
    fn new() -> Self {
        Self {
            inner: Session::new(),
        }
    }

    /// What the client is looking at now, under measures `p` and `q`: the
    /// view as (node, top digits, ln height, rows, right), and the path as
    /// (node, branches). ValueError for a demand the tree cannot take.
    #[pyo3(signature = (tree, p, q, view=None, path=None))]
    fn demand(
        &mut self,
        tree: PyRef<'_, PythonTree>,
        p: usize,
        q: usize,
        view: Option<(u64, Vec<u8>, f64, f64, f64)>,
        path: Option<(u64, Vec<u32>)>,
    ) -> PyResult<()> {
        let view = view.map(|(node, top, ln_height, rows, right)| View {
            node,
            top,
            ln_height,
            rows,
            right,
        });
        let path = path.map(|(node, branches)| Path { node, branches });
        let demand = Demand {
            lens: Lens { p, q },
            view,
            path,
        };
        self.inner.demand(&tree.inner, demand).map_err(error)
    }

    /// Read on, and what is new to the client: ("answer", measure, node,
    /// parent, branch, runs, count) with runs as (rank, branches,
    /// ln_weights, top)
    /// and parent, branch and count None where there are none;
    /// ("unknown", node); ("placed", node, branches); ("lacks", measure,
    /// node); ("failed", measure, node, message).
    fn step(
        &mut self,
        mut tree: PyRefMut<'_, PythonTree>,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let session = &mut self.inner;
        let tree = &mut tree.inner;
        // Without the GIL, so the model's thread runs while views are read.
        let updates = py.detach(|| session.step(tree)).map_err(error)?;
        updates
            .into_iter()
            .map(|update| {
                Ok(match update {
                    Update::Answer {
                        measure,
                        node,
                        edge,
                        runs,
                        count,
                    } => {
                        let runs: Vec<_> = runs.iter().map(run).collect();
                        let (parent, branch) = edge.unzip();
                        ("answer", measure, node, parent, branch, runs, count)
                            .into_pyobject(py)?
                            .into_any()
                            .unbind()
                    }
                    Update::Unknown(node) => {
                        ("unknown", node).into_pyobject(py)?.into_any().unbind()
                    }
                    Update::Placed(Path { node, branches }) => ("placed", node, branches)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                    Update::Lacks { measure, node } => ("lacks", measure, node)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                    Update::Failed {
                        measure,
                        node,
                        message,
                    } => ("failed", measure, node, message)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                })
            })
            .collect()
    }

    /// The distributions waited on, the most valuable first, as (measure,
    /// node, worth, ahead): ahead the branches still to place below a node
    /// on the path.
    fn blocked(&self) -> Vec<(usize, u64, f64, Vec<u32>)> {
        self.inner
            .blocked()
            .into_iter()
            .map(|blocked| (blocked.measure, blocked.node, blocked.worth, blocked.ahead))
            .collect()
    }

    /// The source will never have this distribution.
    fn lacks(&mut self, measure: usize, node: u64) {
        self.inner.lacks(measure, node);
    }

    /// Computing this distribution failed.
    fn failed(&mut self, measure: usize, node: u64, message: &str) {
        self.inner.failed(measure, node, message);
    }
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    module.add_class::<PythonSession>()?;
    Ok(())
}
