use serde::de::DeserializeOwned;
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::distribution::Run;
use crate::layout::{self, Camera, Frame, Viewport};
use crate::tree::{Advance, Lens, Read, Tree as NativeTree, Walk};

fn js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

fn from_js<T: DeserializeOwned>(value: JsValue) -> std::result::Result<T, JsError> {
    serde_wasm_bindgen::from_value(value).map_err(js_error)
}

fn to_js<T: Serialize + ?Sized>(value: &T) -> std::result::Result<JsValue, JsError> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true);
    value.serialize(&serializer).map_err(js_error)
}

/// One message of an Expand answer: what it places of a node's children. The
/// client names the node.
#[derive(Serialize)]
struct Message<'a> {
    runs: Vec<RunMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    children: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunMessage<'a> {
    rank: u32,
    branches: &'a [u32],
    ln_below: &'a [f64],
}

/// The integer-sequence tree, its distributions and their geometry, owned by
/// Rust.
#[wasm_bindgen]
pub struct Tree {
    pub(crate) inner: NativeTree,
    frames: Rc<RefCell<Vec<Frame>>>,
}

#[wasm_bindgen]
impl Tree {
    #[wasm_bindgen(constructor)]
    pub fn new(measures: usize) -> Self {
        Self {
            inner: NativeTree::new(measures),
            frames: Rc::default(),
        }
    }

    pub fn root(&self) -> u64 {
        crate::tree::ROOT
    }

    /// Get or create a child, returning its id: the hash every server shares.
    pub fn child(&mut self, parent: u64, branch: u32) -> std::result::Result<u64, JsError> {
        self.inner.child(parent, branch).map_err(js_error)
    }

    pub fn descend(&mut self, node: u64, path: &[u32]) -> std::result::Result<u64, JsError> {
        self.inner.descend(node, path).map_err(js_error)
    }

    pub fn parent(&self, id: u64) -> std::result::Result<Option<u64>, JsError> {
        self.inner.parent(id).map_err(js_error)
    }

    /// Branch from the parent, or -1 for the root. A number safely represents every u32 branch.
    pub fn branch(&self, id: u64) -> std::result::Result<f64, JsError> {
        Ok(self
            .inner
            .branch(id)
            .map_err(js_error)?
            .map_or(-1.0, f64::from))
    }

    /// Branches from the root to `id`, as a Uint32Array.
    pub fn path(&self, id: u64) -> std::result::Result<Vec<u32>, JsError> {
        self.inner.path(id).map_err(js_error)
    }

    /// Node ids from `origin` through `id`, as a BigUint64Array.
    pub fn chain(&self, origin: u64, id: u64) -> std::result::Result<Vec<u64>, JsError> {
        self.inner.chain(origin, id).map_err(js_error)
    }

    /// Add one answer for a node, validated here: its runs packed flat, each
    /// starting at `ranks[i]` with `lengths[i]` branches and one edge more,
    /// and the node's end where the source knows it. Whether anything
    /// changed. Typed arrays cross as copies, so a whole vocabulary never
    /// becomes JavaScript objects.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        id: u64,
        measure: usize,
        ranks: Vec<u32>,
        lengths: Vec<u32>,
        branches: Vec<u32>,
        ln_below: Vec<f64>,
        children: Option<u32>,
    ) -> std::result::Result<bool, JsError> {
        let (mut branch, mut edge) = (0, 0);
        let mut runs = Vec::with_capacity(ranks.len() + 1);
        for (&rank, &length) in ranks.iter().zip(&lengths) {
            let length = length as usize;
            let (Some(run_branches), Some(run_edges)) = (
                branches.get(branch..branch + length),
                ln_below.get(edge..edge + length + 1),
            ) else {
                return Err(JsError::new(
                    "runs pack fewer branches or edges than their lengths",
                ));
            };
            runs.push(Run::new(rank, run_branches.to_vec(), run_edges.to_vec()).map_err(js_error)?);
            (branch, edge) = (branch + length, edge + length + 1);
        }
        runs.extend(children.map(Run::end_at));
        self.inner.insert(id, measure, runs).map_err(js_error)
    }

    /// Whether `measure` has said anything of the node's children.
    pub fn has(&self, id: u64, measure: usize) -> std::result::Result<bool, JsError> {
        Ok(self
            .inner
            .distribution(id, measure)
            .map_err(js_error)?
            .is_some())
    }

    /// Whether `measure` says the node has no children at all.
    pub fn ended(&self, id: u64, measure: usize) -> std::result::Result<bool, JsError> {
        Ok(self
            .inner
            .distribution(id, measure)
            .map_err(js_error)?
            .is_some_and(|d| d.ended()))
    }

    /// Where p places the child along `branch` in `id`, as [rank, p mass
    /// before it, ln p, −ln q (infinite while unknown)]; empty where p does
    /// not place it.
    pub fn entry(
        &self,
        id: u64,
        p: usize,
        q: usize,
        branch: u32,
    ) -> std::result::Result<Vec<f64>, JsError> {
        let entry = self
            .inner
            .entry(id, Lens { p, q }, branch)
            .map_err(js_error)?;
        Ok(entry.map_or_else(Vec::new, |e| {
            vec![f64::from(e.rank), e.before, e.ln_p, e.nats_q]
        }))
    }

    pub fn statistics(&self, id: u64, p: usize, q: usize) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.statistics(id, Lens { p, q }).map_err(js_error)?)
    }

    /// What this tree, as a recording, answers to an Expand reading `read`
    /// below `node`: the messages it can give before it lacks a distribution.
    pub fn answer(
        &mut self,
        node: u64,
        measure: usize,
        read: JsValue,
    ) -> std::result::Result<JsValue, JsError> {
        let read: Read = from_js(read)?;
        let mut answers = Vec::new();
        if self.inner.contains(node) {
            let mut walk = Walk::new(node, read).map_err(js_error)?;
            while let Advance::Children(_, runs, children) =
                walk.advance(&mut self.inner, measure).map_err(js_error)?
            {
                answers.push((runs, children));
            }
        }
        let messages: Vec<_> = answers
            .iter()
            .map(|(runs, children)| Message {
                runs: runs
                    .iter()
                    .map(|run| RunMessage {
                        rank: run.rank(),
                        branches: run.branches(),
                        ln_below: run.ln_below(),
                    })
                    .collect(),
                children: *children,
            })
            .collect();
        to_js(&messages)
    }
}

#[wasm_bindgen(js_name = layout)]
pub fn layout_frame(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
    view: JsValue,
    merge_px: f64,
) -> std::result::Result<LayoutFrame, JsError> {
    let camera: Camera = from_js(camera)?;
    let view: Viewport = from_js(view)?;
    let frame = tree.frames.borrow_mut().pop().unwrap_or_default();
    Ok(LayoutFrame {
        inner: layout::layout_into(&tree.inner, Lens { p, q }, camera, view, merge_px, frame)
            .map_err(js_error)?,
        pool: Rc::clone(&tree.frames),
    })
}

/// A frame stays in Wasm; only visible boxes and requests cross.
#[wasm_bindgen]
pub struct LayoutFrame {
    inner: Frame,
    pool: Rc<RefCell<Vec<Frame>>>,
}

impl Drop for LayoutFrame {
    fn drop(&mut self) {
        let mut frame = std::mem::take(&mut self.inner);
        frame.clear();
        let mut pool = self.pool.borrow_mut();
        if pool.len() < 2 {
            pool.push(frame);
        }
    }
}

#[wasm_bindgen]
impl LayoutFrame {
    /// Each visible box's node, parallel to `boxes`.
    pub fn nodes(&self) -> Vec<u64> {
        self.inner.boxes.iter().map(|b| b.node).collect()
    }

    /// Eleven f64s per visible box; ranks and counts remain exact.
    pub fn boxes(&self) -> Vec<f64> {
        let mut packed = Vec::with_capacity(self.inner.boxes.len() * 11);
        for b in &self.inner.boxes {
            packed.extend_from_slice(&[
                f64::from(b.rank),
                f64::from(b.count),
                match b.kind {
                    layout::BoxKind::Node => 0.0,
                    layout::BoxKind::Leaf => 1.0,
                    layout::BoxKind::Gap => 2.0,
                    layout::BoxKind::Merged => 3.0,
                },
                b.branch.map_or(-1.0, f64::from),
                b.p,
                b.q,
                b.ln_mass,
                b.x0,
                b.x1,
                b.y0,
                b.y1,
            ]);
        }
        packed
    }

    pub fn misses(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.misses)
    }

    #[wasm_bindgen(js_name = originTop)]
    pub fn origin_top(&self) -> f64 {
        self.inner.origin_top
    }

    #[wasm_bindgen(js_name = originBottom)]
    pub fn origin_bottom(&self) -> f64 {
        self.inner.origin_bottom
    }

    pub fn cover(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.cover)
    }

    pub fn reanchor(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&layout::reanchor(&self.inner))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pick(
        &self,
        tree: &Tree,
        p: usize,
        q: usize,
        camera: JsValue,
        view: JsValue,
        x_px: f64,
        y_px: f64,
        pixel_px: f64,
    ) -> std::result::Result<JsValue, JsError> {
        let camera: Camera = from_js(camera)?;
        let view: Viewport = from_js(view)?;
        let picked = layout::pick(
            &tree.inner,
            Lens { p, q },
            &self.inner,
            camera,
            view,
            x_px,
            y_px,
            pixel_px,
        )
        .map_err(js_error)?;
        to_js(&picked)
    }
}

/// The camera's anchor, or the deepest edge above it that p still places.
#[wasm_bindgen]
pub fn repair(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::repair(&tree.inner, Lens { p, q }, camera).map_err(js_error)?)
}

#[wasm_bindgen]
pub fn extent(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::extent(&tree.inner, Lens { p, q }, camera).map_err(js_error)?)
}

/// Return the clamped camera `a`; the input camera is not mutated.
#[wasm_bindgen(js_name = clampPosition)]
pub fn clamp_position(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<f64, JsError> {
    let camera: Camera = from_js(camera)?;
    layout::clamp_position(&tree.inner, Lens { p, q }, camera).map_err(js_error)
}
