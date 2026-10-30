use serde::de::DeserializeOwned;
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::codec;
use crate::layout::{self, Boundary, Camera, Frame, Viewport};
use crate::tree::{Lens, Tree as NativeTree};

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

/// A nested interval tree with listings and slot views owned by Rust.
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

    pub fn root(&self) -> u32 {
        self.inner.root()
    }

    /// Get or create a child, returning its stable numeric node id.
    pub fn child(&mut self, parent: u32, label: u32) -> std::result::Result<u32, JsError> {
        self.inner.child(parent, label).map_err(js_error)
    }

    pub fn parent(&self, id: u32) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.parent(id).map_err(js_error)?)
    }

    /// Child edge label, or -1 for the root. A number safely represents every u32 label.
    pub fn label(&self, id: u32) -> std::result::Result<f64, JsError> {
        Ok(self
            .inner
            .label(id)
            .map_err(js_error)?
            .map_or(-1.0, f64::from))
    }

    /// Edge labels from the root to `id`, as a Uint32Array.
    pub fn path(&self, id: u32) -> std::result::Result<Vec<u32>, JsError> {
        self.inner.path(id).map_err(js_error)
    }

    /// Node ids from the root through `id`, as a plain number array.
    pub fn chain(&self, id: u32) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.chain(id).map_err(js_error)?)
    }

    #[wasm_bindgen(js_name = setListing)]
    pub fn set_listing(
        &mut self,
        id: u32,
        measure: usize,
        listing: JsValue,
    ) -> std::result::Result<(), JsError> {
        self.inner
            .set_listing(id, measure, from_js(listing)?)
            .map_err(js_error)
    }

    pub fn listing(&self, id: u32, measure: usize) -> std::result::Result<JsValue, JsError> {
        match self.inner.listing(id, measure).map_err(js_error)? {
            Some(listing) => to_js(listing),
            None => Ok(JsValue::NULL),
        }
    }

    pub fn expanded(&self, id: u32, measure: usize) -> std::result::Result<bool, JsError> {
        self.inner.expanded(id, measure).map_err(js_error)
    }

    #[wasm_bindgen(js_name = hasListing)]
    pub fn has_listing(&self, id: u32, measure: usize) -> std::result::Result<bool, JsError> {
        Ok(self.inner.listing(id, measure).map_err(js_error)?.is_some())
    }

    /// Scalar queries never copy a model's vocabulary into JavaScript.
    pub fn horizon(&self, id: u32, measure: usize) -> std::result::Result<f64, JsError> {
        Ok(self
            .inner
            .listing(id, measure)
            .map_err(js_error)?
            .map_or(-1.0, |v| v.horizon))
    }

    pub fn complete(&self, id: u32, measure: usize) -> std::result::Result<f64, JsError> {
        Ok(self
            .inner
            .listing(id, measure)
            .map_err(js_error)?
            .map_or(f64::INFINITY, |v| v.complete))
    }

    #[wasm_bindgen(js_name = slotOf)]
    pub fn slot_of(
        &self,
        id: u32,
        p: usize,
        q: usize,
        label: u32,
    ) -> std::result::Result<f64, JsError> {
        Ok(self
            .inner
            .slots(id, Lens { p, q })
            .map_err(js_error)?
            .slot_of(label)
            .map_or(-1.0, |v| v as f64))
    }

    pub fn statistics(&self, id: u32, p: usize, q: usize) -> std::result::Result<JsValue, JsError> {
        to_js(
            &self
                .inner
                .slots(id, Lens { p, q })
                .map_err(js_error)?
                .statistics(p == q),
        )
    }

    #[wasm_bindgen(js_name = qBound)]
    pub fn q_bound(
        &self,
        id: u32,
        p: usize,
        q: usize,
        slot: usize,
    ) -> std::result::Result<bool, JsError> {
        Ok(self
            .inner
            .slots(id, Lens { p, q })
            .map_err(js_error)?
            .q_bound(slot))
    }

    pub fn slots(&self, id: u32, p: usize, q: usize) -> std::result::Result<JsValue, JsError> {
        let slots = self.inner.slots(id, Lens { p, q }).map_err(js_error)?;
        to_js(&slots.view())
    }
}

#[wasm_bindgen(js_name = decodeListing)]
pub fn decode_listing(
    horizon: f64,
    complete: f64,
    edges: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let edges: Vec<codec::WireEdge> = from_js(edges)?;
    to_js(&codec::decode(&edges, horizon, complete))
}

#[wasm_bindgen(js_name = encodeListing)]
pub fn encode_listing(
    labels: Vec<u32>,
    ln_weights: Vec<f64>,
    horizon: f64,
) -> std::result::Result<JsValue, JsError> {
    if labels.len() != ln_weights.len() {
        return Err(JsError::new(
            "labels and lnWeights must have the same length",
        ));
    }
    to_js(&codec::encode(&labels, &ln_weights, horizon))
}

#[allow(clippy::too_many_arguments)]
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

/// A frame stays in Wasm. Picking and reanchoring borrow it instead of
/// round-tripping every known (usually subpixel) branch through JS objects.
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
    /// Twelve f64s per visible box; numeric ids and integer slots remain exact.
    pub fn boxes(&self) -> Vec<f64> {
        let mut packed = Vec::with_capacity(self.inner.boxes.len() * 12);
        for b in &self.inner.boxes {
            packed.extend_from_slice(&[
                f64::from(b.node),
                b.slot as f64,
                b.count as f64,
                match b.kind {
                    layout::BoxKind::Node => 0.0,
                    layout::BoxKind::Leaf => 1.0,
                    layout::BoxKind::Tail => 2.0,
                    layout::BoxKind::Merged => 3.0,
                },
                b.label.map_or(-1.0, f64::from),
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

    pub fn known(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.known.materialize())
    }
    pub fn walked(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.walked)
    }
    pub fn unexpanded(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.unexpanded)
    }
    pub fn boundaries(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.boundaries)
    }
    #[wasm_bindgen(js_name = rootTop)]
    pub fn root_top(&self) -> f64 {
        self.inner.root_top
    }
    #[wasm_bindgen(js_name = rootBottom)]
    pub fn root_bottom(&self) -> f64 {
        self.inner.root_bottom
    }
    pub fn cover(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&self.inner.cover)
    }
    pub fn reanchor(&self) -> std::result::Result<JsValue, JsError> {
        to_js(&layout::reanchor(&self.inner))
    }

    pub fn pick(
        &self,
        x: f64,
        k: f64,
        width: f64,
        height: f64,
        x_px: f64,
        y_px: f64,
    ) -> std::result::Result<JsValue, JsError> {
        let cam = Camera {
            x,
            k,
            anchor: Boundary { node: 0, slot: 0 },
            a: 0.0,
        };
        to_js(&layout::pick(
            &self.inner,
            cam,
            Viewport { width, height },
            x_px,
            y_px,
        ))
    }
}

#[wasm_bindgen(js_name = normalizeBoundary)]
pub fn normalize_boundary(
    tree: &Tree,
    p: usize,
    q: usize,
    boundary: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let boundary: Boundary = from_js(boundary)?;
    to_js(&layout::normalize(&tree.inner, Lens { p, q }, boundary).map_err(js_error)?)
}

/// Return a reachable anchor without mutating the supplied camera.
#[wasm_bindgen(js_name = repairBoundary)]
pub fn repair_boundary(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::repair(&tree.inner, Lens { p, q }, camera.anchor).map_err(js_error)?)
}

#[wasm_bindgen(js_name = place)]
pub fn place_node(
    tree: &Tree,
    p: usize,
    q: usize,
    id: u32,
) -> std::result::Result<JsValue, JsError> {
    to_js(&layout::place(&tree.inner, Lens { p, q }, id).map_err(js_error)?)
}

#[wasm_bindgen(js_name = rootExtent)]
pub fn root_extent(
    tree: &Tree,
    p: usize,
    q: usize,
    camera: JsValue,
) -> std::result::Result<JsValue, JsError> {
    let camera: Camera = from_js(camera)?;
    to_js(&layout::root_extent(&tree.inner, Lens { p, q }, camera).map_err(js_error)?)
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
