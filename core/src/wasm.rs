use serde::de::DeserializeOwned;
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::client::ClientTree;
use crate::distribution::Run;
use crate::layout::{self, Anchor, Camera, Frame, Viewport};
use crate::session::{Demand, Path, Session as NativeSession, Update, View};
use crate::tree::{child_hash, Lens, Tree as NativeTree};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(message: &str);
}

/// A panic traps as "unreachable", saying nothing, and leaves whatever it
/// borrowed borrowed for good: every later call reports aliasing instead.
/// Its message and place, said first, name the actual fault.
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| error(&info.to_string())));
}

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

/// A session's update, as the page reads it: node ids as BigInts.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum UpdateMessage<'a> {
    Answer {
        measure: usize,
        node: u128,
        parent: Option<u128>,
        branch: Option<u32>,
        runs: Vec<RunMessage<'a>>,
        count: Option<u32>,
    },
    Unknown {
        node: u128,
    },
    Placed {
        node: u128,
        branches: &'a [u32],
    },
    Lacks {
        measure: usize,
        node: u128,
    },
    Failed {
        measure: usize,
        node: u128,
        message: &'a str,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunMessage<'a> {
    rank: u32,
    branches: &'a [u32],
    ln_weights: &'a [f64],
    #[serde(with = "serde_bytes_slice")]
    top: Vec<u8>,
}

/// A run's top crosses as a Uint8Array.
mod serde_bytes_slice {
    pub fn serialize<S: serde::Serializer>(
        digits: &[u8],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(digits)
    }
}

fn update_message(update: &Update) -> UpdateMessage<'_> {
    match update {
        Update::Answer {
            measure,
            node,
            edge,
            runs,
            count,
        } => UpdateMessage::Answer {
            measure: *measure,
            node: u128::from(*node),
            parent: edge.map(|(parent, _)| u128::from(parent)),
            branch: edge.map(|(_, branch)| branch),
            runs: runs
                .iter()
                .map(|run| RunMessage {
                    rank: run.rank(),
                    branches: run.branches(),
                    ln_weights: run.ln_weights(),
                    top: run.top_digits(),
                })
                .collect(),
            count: *count,
        },
        Update::Unknown(node) => UpdateMessage::Unknown {
            node: u128::from(*node),
        },
        Update::Placed(Path { node, branches }) => UpdateMessage::Placed {
            node: u128::from(*node),
            branches,
        },
        Update::Lacks { measure, node } => UpdateMessage::Lacks {
            measure: *measure,
            node: u128::from(*node),
        },
        Update::Failed {
            measure,
            node,
            message,
        } => UpdateMessage::Failed {
            measure: *measure,
            node: u128::from(*node),
            message,
        },
    }
}

/// The integer-sequence tree, its distributions and their geometry, owned by
/// Rust.
#[wasm_bindgen]
pub struct Tree {
    pub(crate) inner: ClientTree,
    frames: Rc<RefCell<Vec<Frame>>>,
}

#[wasm_bindgen]
impl Tree {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: ClientTree::new(),
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
    /// starting at `ranks[i]` with `lengths[i]` branches and weights and a
    /// top of `digits[i]` digits, and the node's end where the source knows
    /// it. Whether anything changed. Typed arrays cross as copies, so a
    /// whole vocabulary never becomes JavaScript objects.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        id: u64,
        measure: usize,
        ranks: Vec<u32>,
        lengths: Vec<u32>,
        branches: Vec<u32>,
        ln_weights: Vec<f64>,
        digits: Vec<u32>,
        tops: Vec<u8>,
        children: Option<u32>,
    ) -> std::result::Result<bool, JsError> {
        let runs = unpack_runs(ranks, lengths, branches, ln_weights, digits, tops)?;
        self.inner
            .insert(id, measure, runs, children)
            .map_err(js_error)
    }

    /// Whether `measure` has said anything of the node's children.
    pub fn has(&self, id: u64, measure: usize) -> std::result::Result<bool, JsError> {
        Ok(self.inner.distribution(id, measure).is_some())
    }

    /// Whether `measure` says the node has no children at all.
    pub fn ended(&self, id: u64, measure: usize) -> std::result::Result<bool, JsError> {
        Ok(self
            .inner
            .distribution(id, measure)
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
        let entry = self.inner.entry(id, Lens { p, q }, branch);
        Ok(entry.map_or_else(Vec::new, |e| {
            vec![f64::from(e.rank), e.before, e.ln_p, e.nats_q]
        }))
    }

    pub fn statistics(&self, id: u64, p: usize, q: usize) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.statistics(id, Lens { p, q }))
    }
}

/// The lossless source for a snapshot session.
#[wasm_bindgen]
pub struct RecordedTree {
    inner: NativeTree,
}

#[wasm_bindgen]
impl RecordedTree {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: NativeTree::new(),
        }
    }

    pub fn child(&mut self, parent: u64, branch: u32) -> std::result::Result<u64, JsError> {
        self.inner.child(parent, branch).map_err(js_error)
    }

    pub fn descend(&mut self, node: u64, path: &[u32]) -> std::result::Result<u64, JsError> {
        self.inner.descend(node, path).map_err(js_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        id: u64,
        measure: usize,
        ranks: Vec<u32>,
        lengths: Vec<u32>,
        branches: Vec<u32>,
        ln_weights: Vec<f64>,
        digits: Vec<u32>,
        tops: Vec<u8>,
        children: Option<u32>,
    ) -> std::result::Result<bool, JsError> {
        let runs = unpack_runs(ranks, lengths, branches, ln_weights, digits, tops)?;
        self.inner
            .insert(id, measure, runs, children)
            .map_err(js_error)
    }
}

fn unpack_runs(
    ranks: Vec<u32>,
    lengths: Vec<u32>,
    branches: Vec<u32>,
    ln_weights: Vec<f64>,
    digits: Vec<u32>,
    tops: Vec<u8>,
) -> std::result::Result<Vec<Run>, JsError> {
    if ranks.len() != lengths.len() || ranks.len() != digits.len() {
        return Err(JsError::new("each run needs a rank, length and precision"));
    }
    let (mut branches, mut weights, mut tops) =
        (branches.as_slice(), ln_weights.as_slice(), tops.as_slice());
    let mut runs = Vec::with_capacity(ranks.len());
    for ((rank, length), count) in ranks.into_iter().zip(lengths).zip(digits) {
        let (
            Some((run_branches, rest_branches)),
            Some((run_weights, rest_weights)),
            Some((top, rest_top)),
        ) = (
            branches.split_at_checked(length as usize),
            weights.split_at_checked(length as usize),
            tops.split_at_checked(count as usize),
        )
        else {
            return Err(JsError::new(
                "runs pack fewer branches, weights or digits than their lengths",
            ));
        };
        runs.push(Run::from_slices(rank, run_branches, run_weights, top).map_err(js_error)?);
        (branches, weights, tops) = (rest_branches, rest_weights, rest_top);
    }
    if !branches.is_empty() || !weights.is_empty() || !tops.is_empty() {
        return Err(JsError::new(
            "runs pack more branches, weights or digits than their lengths",
        ));
    }
    Ok(runs)
}

/// A client's session with a recording: what the page looks at, read through
/// the recorded tree. A recording has all it will ever have.
#[wasm_bindgen]
pub struct Session {
    inner: NativeSession,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: NativeSession::new(),
        }
    }

    /// What the page looks at now: a view and a path, each or null.
    pub fn demand(
        &mut self,
        recorded: &RecordedTree,
        p: usize,
        q: usize,
        view: JsValue,
        path: JsValue,
    ) -> std::result::Result<(), JsError> {
        let view: Option<View> = from_js(view)?;
        let path: Option<Path> = from_js(path)?;
        let demand = Demand {
            lens: Lens { p, q },
            view,
            path,
        };
        self.inner.demand(&recorded.inner, demand).map_err(js_error)
    }

    /// Everything new to the page: whatever the recording does not hold is
    /// said to be lacking, and read past.
    pub fn answer(&mut self, recorded: &mut RecordedTree) -> std::result::Result<JsValue, JsError> {
        let mut updates = Vec::new();
        loop {
            updates.extend(self.inner.step(&mut recorded.inner).map_err(js_error)?);
            let blocked = self.inner.blocked();
            if blocked.is_empty() {
                break;
            }
            for missing in blocked {
                self.inner.lacks(missing.measure, missing.node);
            }
        }
        let messages: Vec<_> = updates.iter().map(update_message).collect();
        to_js(&messages)
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_name = layout)]
#[allow(clippy::too_many_arguments)]
pub fn layout_frame(
    tree: &Tree,
    p: usize,
    q: usize,
    origin: u64,
    x: f64,
    k: f64,
    a: f64,
    path: Vec<u32>,
    digits: Vec<u8>,
    width: f64,
    height: f64,
    row_px: f64,
) -> std::result::Result<LayoutFrame, JsError> {
    let camera = Camera {
        origin,
        x,
        k,
        a,
        anchor: Anchor { path, digits },
    };
    let view = Viewport { width, height };
    let frame = tree.frames.borrow_mut().pop().unwrap_or_default();
    Ok(LayoutFrame {
        inner: layout::layout_into(&tree.inner, Lens { p, q }, &camera, view, row_px, frame)
            .map_err(js_error)?,
        pool: Rc::clone(&tree.frames),
        projection: None,
    })
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn project(
    tree: &Tree,
    p: usize,
    q: usize,
    origin: u64,
    x: f64,
    k: f64,
    a: f64,
    path: Vec<u32>,
    digits: Vec<u8>,
    width: f64,
    height: f64,
    pixel_ratio: f64,
) -> std::result::Result<LayoutFrame, JsError> {
    let mut camera = Camera {
        origin,
        x,
        k,
        a,
        anchor: Anchor { path, digits },
    };
    let viewport = Viewport { width, height };
    let frame = tree.frames.borrow_mut().pop().unwrap_or_default();
    let (inner, demand) = layout::project(
        &tree.inner,
        Lens { p, q },
        &mut camera,
        viewport,
        pixel_ratio,
        frame,
    )
    .map_err(js_error)?;
    Ok(LayoutFrame {
        inner,
        pool: Rc::clone(&tree.frames),
        projection: Some(ProjectedView {
            anchor: camera.anchor,
            a: camera.a,
            demand,
        }),
    })
}

#[derive(Serialize)]
struct ProjectedView {
    anchor: Anchor,
    a: f64,
    demand: View,
}

/// A frame stays in Wasm; only visible boxes cross.
#[wasm_bindgen]
pub struct LayoutFrame {
    inner: Frame,
    pool: Rc<RefCell<Vec<Frame>>>,
    projection: Option<ProjectedView>,
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
    #[wasm_bindgen(js_name = takeProjection)]
    pub fn take_projection(&mut self) -> std::result::Result<JsValue, JsError> {
        to_js(
            &self
                .projection
                .take()
                .ok_or_else(|| js_error("frame has no projection"))?,
        )
    }

    /// Each visible box's node and its child's hash, 0 for a gap's second id.
    /// The branch column in `boxes` distinguishes a gap from a child.
    #[wasm_bindgen(js_name = boxNodes)]
    pub fn box_nodes(&self) -> Vec<u64> {
        self.inner
            .boxes
            .iter()
            .flat_map(|b| {
                [
                    b.node,
                    b.branch.map_or(0, |branch| child_hash(b.node, branch)),
                ]
            })
            .collect()
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

    /// The known children of `node` that reach into the view, however thin:
    /// three f64s each, its branch and its top and bottom in view heights.
    pub fn known(
        &self,
        tree: &Tree,
        p: usize,
        q: usize,
        node: u64,
    ) -> std::result::Result<Vec<f64>, JsError> {
        let known =
            layout::known(&tree.inner, Lens { p, q }, &self.inner, node).map_err(js_error)?;
        Ok(known
            .into_iter()
            .flat_map(|(branch, y0, y1)| [f64::from(branch), y0, y1])
            .collect())
    }

    /// Each span's node, parallel to `spans`.
    #[wasm_bindgen(js_name = spanNodes)]
    pub fn span_nodes(&self) -> Vec<u64> {
        self.inner.spans.iter().map(|span| span.node).collect()
    }

    /// Five f64s per span: its rows' top and bottom in view heights, and
    /// where its children start, its fill ends and its outline reaches.
    pub fn spans(&self) -> Vec<f64> {
        self.inner
            .spans
            .iter()
            .flat_map(|span| [span.y0, span.y1, span.x0, span.fill, span.reach])
            .collect()
    }

    #[wasm_bindgen(js_name = originTop)]
    pub fn origin_top(&self) -> f64 {
        self.inner.origin_top
    }

    #[wasm_bindgen(js_name = originBottom)]
    pub fn origin_bottom(&self) -> f64 {
        self.inner.origin_bottom
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rasterize(
        &self,
        tree: &Tree,
        p: usize,
        q: usize,
        cam_x: f64,
        cam_k: f64,
        width_px: usize,
        height_px: usize,
        dpr: f64,
        surface_rgb: u32,
        tile_rgb: u32,
        stroke_rgb: u32,
        out: &mut [u8],
    ) {
        crate::raster::rasterize(
            &tree.inner,
            Lens { p, q },
            &self.inner,
            cam_x,
            cam_k,
            width_px,
            height_px,
            dpr,
            surface_rgb,
            tile_rgb,
            stroke_rgb,
            out,
        );
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
            &camera,
            view,
            x_px,
            y_px,
            pixel_px,
        )
        .map_err(js_error)?;
        // Ancestor ids are u64s, as are the ids in boxNodes.
        picked
            .serialize(
                &serde_wasm_bindgen::Serializer::new()
                    .serialize_missing_as_null(true)
                    .serialize_large_number_types_as_bigints(true),
            )
            .map_err(js_error)
    }
}

/// What the camera looks at, as a session's view of `rows` pixel rows.
#[wasm_bindgen]
pub fn view(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
    viewport: JsValue,
    rows: f64,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    let viewport: Viewport = from_js(viewport)?;
    to_js(&layout::view(&tree.inner, Lens { p, q }, &camera, viewport, rows).map_err(js_error)?)
}

/// The camera's point under the name that keeps the view exact.
#[wasm_bindgen]
pub fn settle(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::settle(&tree.inner, Lens { p, q }, &camera).map_err(js_error)?)
}

/// The camera's point moved to what the view shows `fy` view heights down,
/// and where it then is.
#[wasm_bindgen]
pub fn aim(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
    fy: f64,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::aim(&tree.inner, Lens { p, q }, &camera, fy).map_err(js_error)?)
}

#[wasm_bindgen]
pub fn extent(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
    depth: usize,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::extent(&tree.inner, Lens { p, q }, &camera, depth).map_err(js_error)?)
}

/// Where the node `path` leads to from `origin` sits, as layout places it.
#[wasm_bindgen]
pub fn reach(
    tree: &Tree,
    p: usize,
    q: usize,
    origin: u64,
    path: Vec<u32>,
) -> std::result::Result<JsValue, JsError> {
    to_js(&layout::reach(&tree.inner, Lens { p, q }, origin, &path).map_err(js_error)?)
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
    layout::clamp_position(&tree.inner, Lens { p, q }, &camera).map_err(js_error)
}
