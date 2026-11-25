//! Python adapter for servers: node identity, distributions and what each
//! client's view lacks stay in the core; a server only computes the
//! distributions its sessions wait on.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::distribution::{log_softmax_offsets, Distribution, Run};
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
        let distributions = py
            .detach(|| packed(rows, Distribution::from_ordered))
            .map_err(error)?;
        slf.borrow_mut().install(nodes, measure, distributions)
    }

    /// A float32 row indexed by branch for each node. Rank ties follow
    /// branch order. Final log widths can be supplied with normalize=False.
    #[pyo3(signature = (nodes, measure, rows, *, heaviest, normalize=true))]
    fn set_rows(
        slf: &Bound<'_, Self>,
        nodes: Vec<u64>,
        measure: usize,
        rows: Vec<PyBuffer<f32>>,
        heaviest: bool,
        normalize: bool,
    ) -> PyResult<Vec<usize>> {
        if nodes.len() != rows.len() {
            return Err(error(format!(
                "{} nodes but {} rows of weights",
                nodes.len(),
                rows.len()
            )));
        }
        let py = slf.py();
        // The buffer exporters can mutate their memory. Give the detached
        // workers owned f32 rows before releasing Python.
        let rows = rows
            .into_iter()
            .map(|row| {
                if row.dimensions() != 1 {
                    return Err(error("each row of weights must be one-dimensional".into()));
                }
                row.to_vec(py)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let distributions = py
            .detach(|| packed(rows, |row| pack_row(row, heaviest, normalize)))
            .map_err(error)?;
        slf.borrow_mut().install(nodes, measure, distributions)
    }

    fn forget(&mut self, node: u64, measure: usize) -> PyResult<()> {
        self.inner.forget(node, measure).map_err(error)
    }
}

fn pack_row(row: Vec<f32>, heaviest: bool, normalize: bool) -> crate::Result<Distribution> {
    let count = u32::try_from(row.len()).map_err(|_| "too many children to rank".to_owned())?;
    let mut children = Vec::with_capacity(row.len());
    for (branch, weight) in (0..count).zip(row) {
        if weight == f32::NEG_INFINITY {
            continue;
        }
        if !weight.is_finite() {
            return Err("a weight must be a log mass".into());
        }
        children.push((branch, weight));
    }
    if heaviest {
        children.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("finite log weights")
                .then_with(|| a.0.cmp(&b.0))
        });
    }
    let (max, ln_sum) = if normalize {
        log_softmax_offsets(children.iter().map(|&(_, weight)| f64::from(weight)))
    } else {
        (0.0, 0.0)
    };
    Distribution::pack_unique(
        children
            .iter()
            .map(|&(branch, weight)| (branch, (f64::from(weight) - max) - ln_sum)),
    )
}

/// Each row's distribution, rows shared out among the cores in order.
fn packed<T: Send>(
    mut rows: Vec<Vec<T>>,
    pack: impl Fn(Vec<T>) -> crate::Result<Distribution> + Sync,
) -> crate::Result<Vec<Distribution>> {
    if rows.len() <= 1 {
        return rows.into_iter().map(pack).collect();
    }
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let share = rows.len().div_ceil(cores);
    let count = rows.len();
    let pack = &pack;
    std::thread::scope(|scope| {
        let workers: Vec<_> = rows
            .chunks_mut(share)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter_mut()
                        .map(|row| pack(std::mem::take(row)))
                        .collect::<crate::Result<Vec<_>>>()
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
    fn install(
        &mut self,
        nodes: Vec<u64>,
        measure: usize,
        distributions: Vec<Distribution>,
    ) -> PyResult<Vec<usize>> {
        for &node in &nodes {
            self.inner.distribution(node, measure).map_err(error)?;
        }
        nodes
            .into_iter()
            .zip(distributions)
            .map(|(node, distribution)| {
                let bytes = distribution.bytes();
                self.inner.set(node, measure, distribution).map_err(error)?;
                Ok(bytes)
            })
            .collect()
    }

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

    /// Forget what was sent of this distribution (e.g. on server cache eviction).
    fn forget(&mut self, measure: usize, node: u64) {
        self.inner.forget(measure, node);
    }
}

#[pymodule]
fn natlas_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonTree>()?;
    module.add_class::<PythonSession>()?;
    Ok(())
}
