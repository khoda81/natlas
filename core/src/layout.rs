//! Anchored geometry. A node's children split its height by p, in p's order,
//! and reach −ln q to the right. A frame lays out what a camera sees and
//! answers picks; what the camera looks at, as a session's view, is told
//! exactly as the frame places it.
//!
//! Positions never come from absolute coordinates. The camera places one
//! point, named by a path of branches below its origin and digits within the
//! node they lead to, and each laid-out node keeps two of its own edges
//! placed exactly in the view and measures everything from the nearer one:
//! a branch 2^-4096 of its parent sits as exactly beside the point as a
//! heavy one.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use std::f64::consts::LN_2;

use crate::client::{ClientMeasure, ClientTree, FixedIndex, IdHasher};
use crate::distribution::{Distribution, Edge, Item, Rank, RankView};
use crate::point::Point;
use crate::pstore::{BranchHit, CDF_SIZE};
use crate::qlayout::{QLayout, WidthIndex};
use crate::session::View;
use crate::tree::{bytes, child_hash, node_id, Branch, Lens, NodeId, Tree, ROOT};
use crate::Result;

/// Read boundary shared by the compact tree and the render tree.
#[doc(hidden)]
pub trait LayoutTree {
    type Store: RankView;

    fn distribution(&self, node: NodeId, measure: usize) -> Result<Option<&Self::Store>>;
    fn nats_q(&self, node: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64>;

    fn packed_q(&self, _node: NodeId, _lens: Lens) -> Option<&Distribution> {
        None
    }

    fn client_measure(&self, _node: NodeId, _measure: usize) -> Option<&ClientMeasure> {
        None
    }
}

impl LayoutTree for Tree {
    type Store = Distribution;

    fn distribution(&self, node: NodeId, measure: usize) -> Result<Option<&Distribution>> {
        self.distribution(node, measure)
    }

    fn packed_q(&self, node: NodeId, lens: Lens) -> Option<&Distribution> {
        if lens.p == lens.q {
            return None;
        }
        self.distribution(node, lens.q)
            .ok()
            .flatten()
            .filter(|q| q.is_packed())
    }

    fn nats_q(&self, node: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64> {
        self.nats_q(node, lens, branch, ln_p)
    }
}

impl LayoutTree for ClientTree {
    type Store = ClientMeasure;

    fn distribution(&self, node: NodeId, measure: usize) -> Result<Option<&ClientMeasure>> {
        Ok(self.distribution(node, measure))
    }

    fn nats_q(&self, node: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64> {
        Ok(self.nats_q(node, lens, branch, ln_p))
    }

    fn client_measure(&self, node: NodeId, measure: usize) -> Option<&ClientMeasure> {
        self.distribution(node, measure)
    }
}

/// Physical pixels within which a coordinate is on a grid line, as the
/// renderer rounds: rows lie on the grid only up to float error.
const ON_GRID: f64 = 1e-6;

/// A point at or below a camera's origin: the node `path` leads to from the
/// origin, and the fraction of that node's p mass above the point, as digits
/// in base 256. No digits is the node's top.
///
/// One point has many such names, a branch of its path standing for the
/// digits that pick that child; [`settle`] chooses among them.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Anchor {
    pub path: Vec<Branch>,
    #[serde(with = "bytes")]
    pub digits: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    /// The node the view is measured from, standing as the root does: x is
    /// q-surprisal from its children's left edge and the view spans e^-x of
    /// its interval. It bounds the view; nothing above it is needed.
    #[serde(with = "node_id")]
    pub origin: NodeId,
    pub x: f64,
    pub k: f64,
    /// A point of the origin...
    pub anchor: Anchor,
    /// ...this many view heights below the top of the view.
    pub a: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BoxKind {
    /// A child whose own children are known.
    Node,
    /// A child whose children are not.
    Leaf,
    /// Children not yet known: they reach past any view until they are.
    Gap,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LayoutBox {
    #[serde(with = "node_id")]
    pub node: NodeId,
    /// The first rank the box covers.
    pub rank: Rank,
    /// Children covered; a gap at a node's bottom has no known count, 0.
    pub count: u32,
    pub kind: BoxKind,
    /// The child's branch; none for a gap or a row.
    pub branch: Option<Branch>,
    /// Conditional probabilities; for a row, geometric means weighted by
    /// what each child covers of it.
    pub p: f64,
    /// NaN while unknown; like p, a representative child weight for a row.
    pub q: f64,
    pub ln_mass: f64,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Extent {
    pub top: f64,
    pub bottom: f64,
}

/// A camera's point named anew, and where it is in the view.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Aim {
    pub anchor: Anchor,
    pub a: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PickResult {
    pub r#box: LayoutBox,
    /// False in null space: the branch is known, its continuation is not.
    pub exact: bool,
    /// Child ids along the picked path through the laid-out part of the frame.
    pub ancestors: Vec<NodeId>,
}

/// Where a laid-out node sits: two of its edges placed exactly in view
/// heights.
#[derive(Clone, Debug)]
pub struct Placed {
    top: (Point, f64),
    bottom: (Point, f64),
    /// ln of how far each of the two is from the view's top, kept beside
    /// them: every edge of the node is measured against both.
    ln_heights: (f64, f64),
    /// ln of the node's p mass below the origin.
    ln_mass: f64,
    /// ln of view heights per unit of the node's mass.
    ln_scale: f64,
    /// Where the node's children start, in nats right of the origin's.
    x: f64,
}

impl Placed {
    fn new(top: (Point, f64), bottom: (Point, f64), ln_mass: f64, ln_scale: f64, x: f64) -> Self {
        let ln_heights = (top.1.abs().ln(), bottom.1.abs().ln());
        Self {
            top,
            bottom,
            ln_heights,
            ln_mass,
            ln_scale,
            x,
        }
    }

    /// The view height of an edge, measured from the placed edge that
    /// tells it best. Outside the two, that is the one it is beyond. Between
    /// them, a height rounds as the larger of the placed edge's own height
    /// and the distance from it, so an edge in view is measured from the one
    /// near it, never from one a great many views away.
    fn y(&self, edge: &Point) -> f64 {
        let (under_top, ln_top) = self.top.0.to(edge);
        if !under_top {
            return self.top.1 - (self.ln_scale + ln_top).exp();
        }
        let (under_bottom, ln_bottom) = self.bottom.0.to(edge);
        if under_bottom || ln_bottom == f64::NEG_INFINITY {
            return self.bottom.1 + (self.ln_scale + ln_bottom).exp();
        }
        let down = self.ln_scale + ln_top;
        let up = self.ln_scale + ln_bottom;
        if down.max(self.ln_heights.0) <= up.max(self.ln_heights.1) {
            self.top.1 + down.exp()
        } else {
            self.bottom.1 - up.exp()
        }
    }

    /// The point at a view height, clamped to the node. Between two edges
    /// both farther than a float tells, the height is somewhere in the one
    /// stretch between them, which holds the whole view: its top stands for
    /// it.
    fn at(&self, y: f64) -> Point {
        let (edge, at) = if y <= self.top.1 || y - self.top.1 <= self.bottom.1 - y {
            &self.top
        } else {
            &self.bottom
        };
        let distance = (y - at).abs();
        if distance == f64::INFINITY {
            return self.top.0.clone();
        }
        edge.moved(*at <= y, distance, -self.ln_scale)
            .unwrap_or_else(|| {
                if *at <= y {
                    Point::bottom()
                } else {
                    Point::top()
                }
            })
    }

    fn quantile(&self) -> Option<QLayout> {
        QLayout::new(
            self.ln_scale,
            (&self.top.0, self.top.1),
            (&self.bottom.0, self.bottom.1),
        )
    }

    /// The node's own extent in view heights.
    fn span(&self) -> (f64, f64) {
        (self.y(&Point::top()), self.y(&Point::bottom()))
    }

    /// Placed within a child of `ln_p` extent spanning `y0..y1`.
    fn child(&self, (y0, y1): (f64, f64), ln_p: f64, x: f64) -> Placed {
        Placed::new(
            (Point::top(), y0),
            (Point::bottom(), y1),
            self.ln_mass + ln_p,
            self.ln_scale + ln_p,
            x,
        )
    }
}

/// View heights from a run's own point within which its edges are placed
/// by f64 offsets from that point: within 2^16 view heights, a float of
/// one is off by at most 2^-37 of a view height. Farther, an edge's height
/// would cancel what places it, so it is measured exactly.
const NEAR: f64 = 65536.0;

/// The view heights of a placed node's edges: each run's own point placed
/// exactly once, and its edges near the view by their f64 offsets from it.
struct Heights<'a, D> {
    placed: &'a Placed,
    d: &'a D,
    quantile: Option<&'a QLayout>,
    /// View heights per unit of the node's mass, where a float holds it.
    scale: f64,
    /// The most recent run's projected base. Edges are read in run order.
    base: Option<(usize, f64)>,
    /// The last edge asked, and its height: the next item's top is the
    /// bottom of the one before.
    last: Option<(Edge, f64)>,
}

impl<'a, D: RankView> Heights<'a, D> {
    fn new(placed: &'a Placed, d: &'a D, quantile: Option<&'a QLayout>) -> Self {
        Self {
            placed,
            d,
            quantile,
            scale: placed.ln_scale.exp(),
            base: None,
            last: None,
        }
    }

    fn y(&mut self, edge: Edge) -> f64 {
        if let Some((last, y)) = self.last {
            if last == edge {
                return y;
            }
        }
        let y = self.measure(edge);
        self.last = Some((edge, y));
        y
    }

    fn measure(&mut self, edge: Edge) -> f64 {
        if let Edge::Of { run, at } = edge {
            if let Some(slice) = self.d.run(run).listed_slice() {
                let base = match self.base {
                    Some((index, at)) if index == run => at,
                    _ => {
                        let at = self.quantile.map_or_else(
                            || self.placed.y(slice.base()),
                            |geometry| geometry.y(slice.q64_base()),
                        );
                        self.base = Some((run, at));
                        at
                    }
                };
                if base.abs() <= NEAR {
                    let offset = slice.offsets()[at];
                    let offset = if self.scale.is_finite() {
                        offset * self.scale
                    } else {
                        (offset.ln() + self.placed.ln_scale).exp()
                    };
                    return base + offset;
                }
            }
        }
        self.placed.y(&self.d.point(edge))
    }

    fn item(&mut self, item: Item) -> (f64, f64) {
        let (top, bottom) = self.d.item_bounds(item);
        let top = self.y(top);
        (top, self.y(bottom))
    }
}

/// Physical rows of one node's children drawn alike, from `y0` to `y1` view
/// heights: from `x0`, where the children start, every pixel wholly inside
/// them up to `fill` is filled, and every pixel from there to `reach`, where
/// the widest of them ends, holds an edge of them and is outline, as is the
/// column at `x0`. Where `fill` is `x0`, an edge of the node's children, or a
/// hole in them, falls in every row: outline throughout.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Span {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub y0: f64,
    pub y1: f64,
    pub x0: f64,
    pub fill: f64,
    pub reach: f64,
}

#[derive(Debug, Default)]
pub struct Frame {
    /// Each child and gap a physical row or more tall: labelled and picked.
    pub boxes: Vec<LayoutBox>,
    /// What is painted: every node's children as spans of rows.
    pub spans: Vec<Span>,
    /// Every node laid out, for picking.
    placed: HashMap<NodeId, Placed, BuildHasherDefault<IdHasher>>,
    /// The shallowest node laid out: every pixel right of where its children
    /// start is in them, so picking walks down from it.
    pub top: NodeId,
    pub origin_top: f64,
    pub origin_bottom: f64,
    /// Reusable per-depth hit buffers for `lay_quantile` (zero allocs when warm).
    hit_pool: Vec<Vec<BranchHit>>,
}

impl Frame {
    /// Empty, keeping the allocations for the next view.
    pub(crate) fn clear(&mut self) {
        self.boxes.clear();
        self.spans.clear();
        self.placed.clear();
        self.top = ROOT;
        self.origin_top = 0.0;
        self.origin_bottom = 0.0;
    }
}

/// A child's or a gap's box, as laid out and as picked.
#[allow(clippy::too_many_arguments)]
fn item_box(
    tree: &impl LayoutTree,
    lens: Lens,
    node: NodeId,
    placed: &Placed,
    d: &impl RankView,
    item: Item,
    (y0, y1): (f64, f64),
    known: Option<f64>,
) -> Result<LayoutBox> {
    let (rank, count, kind, branch, ln_p, nats) = match item {
        Item::Child { run, at } => {
            let run = d.run(run);
            let branch = run.branch(at);
            let ln_p = run.ln_p(at);
            let rank = run.rank() + at as Rank;
            let expanded = tree
                .distribution(child_hash(node, branch), lens.p)?
                .is_some();
            let kind = if expanded {
                BoxKind::Node
            } else {
                BoxKind::Leaf
            };
            let nats = known.map_or_else(|| tree.nats_q(node, lens, branch, ln_p), Ok)?;
            (rank, 1, kind, Some(branch), ln_p, nats)
        }
        Item::Gap { .. } => {
            let rank = d.item_rank(item);
            let end = d.item_end(item);
            let ln_p = known.unwrap_or_else(|| {
                let (top, bottom) = d.item_edges(item);
                Point::ln_between(&top, &bottom)
            });
            (
                rank,
                end.map_or(0, |end| end - rank),
                BoxKind::Gap,
                None,
                ln_p,
                f64::INFINITY,
            )
        }
    };
    let p = ln_p.exp();
    let q = if nats.is_finite() {
        if lens.p == lens.q && branch.is_some() {
            p
        } else {
            (-nats).exp()
        }
    } else {
        f64::NAN
    };
    Ok(LayoutBox {
        node,
        rank,
        count,
        kind,
        branch,
        p,
        q,
        ln_mass: placed.ln_mass + ln_p,
        x0: placed.x,
        x1: placed.x + nats,
        y0,
        y1,
    })
}

struct ChainEntry {
    branch: Branch,
    rank: Rank,
    ln_p: f64,
    nats_q: f64,
}

/// The nodes down a path from an origin, as far as p places its branches:
/// each one's entry in its parent and edges there, and its ln p mass and
/// q-surprisal below the origin.
struct Steps {
    nodes: Vec<NodeId>,
    entries: Vec<ChainEntry>,
    edges: Vec<(Point, Point)>,
    ln_masses: Vec<f64>,
    ln_mass_comps: Vec<f64>,
    xs: Vec<f64>,
}

impl Steps {
    fn push(
        &mut self,
        tree: &impl LayoutTree,
        lens: Lens,
        d: &impl RankView,
        (run, at): (usize, usize),
        (top, bottom): (Point, Point),
    ) -> Result<f64> {
        let i = self.entries.len();
        let node = self.nodes[i];
        let child = d.run(run);
        let branch = child.branch(at);
        let ln_p = child.ln_p(at);
        let nats_q = tree.nats_q(node, lens, branch, ln_p)?;
        // Placed by its extent, sized by its weight.
        let ln_extent = Point::ln_between(&top, &bottom);
        self.nodes.push(child_hash(node, branch));
        self.entries.push(ChainEntry {
            branch,
            rank: d.item_rank(Item::Child { run, at }),
            ln_p,
            nats_q,
        });
        self.edges.push((top, bottom));
        let y = ln_extent - self.ln_mass_comps[i];
        let total = self.ln_masses[i] + y;
        let comp = (total - self.ln_masses[i]) - y;
        self.ln_masses.push(total);
        self.ln_mass_comps.push(comp);
        self.xs.push(self.xs[i] + nats_q);
        Ok(ln_extent)
    }

    fn truncate(&mut self, depth: usize) {
        self.entries.truncate(depth);
        self.edges.truncate(depth);
        self.nodes.truncate(depth + 1);
        self.ln_masses.truncate(depth + 1);
        self.ln_mass_comps.truncate(depth + 1);
        self.xs.truncate(depth + 1);
    }
}

fn steps(tree: &impl LayoutTree, lens: Lens, origin: NodeId, path: &[Branch]) -> Result<Steps> {
    let mut steps = Steps {
        nodes: vec![origin],
        entries: Vec::with_capacity(path.len()),
        edges: Vec::with_capacity(path.len()),
        ln_masses: vec![0.0],
        ln_mass_comps: vec![0.0],
        xs: vec![0.0],
    };
    for (i, &branch) in path.iter().enumerate() {
        let node = steps.nodes[i];
        let Some(p) = tree.distribution(node, lens.p)? else {
            break;
        };
        let Some((run, at)) = p.rank(branch).and_then(|rank| p.locate_rank(rank)) else {
            break;
        };
        steps.push(
            tree,
            lens,
            p,
            (run, at),
            p.item_edges(Item::Child { run, at }),
        )?;
    }
    Ok(steps)
}

/// The known edges nearest a point of a node, above and below it: those of
/// the item holding it, or the node's own.
fn around(d: Option<&impl RankView>, point: &Point) -> (Point, Point) {
    d.and_then(|d| Some(d.item_edges(d.locate(point)?)))
        .filter(|(top, bottom)| point.within(top, bottom))
        .unwrap_or_else(|| (Point::top(), Point::bottom()))
}

/// Bits that tell a point of a node e^`ln_scale` view heights tall to
/// 2^-64 of a view height.
fn precision(ln_scale: f64) -> u64 {
    (64.0 + ln_scale / LN_2).ceil().max(0.0) as u64
}

/// The nodes from the origin down the anchor's path, each one's entry in its
/// parent, where each is placed and its own extent in view heights: the
/// anchor's node around its point, every ancestor at its chain child's edges.
struct Chain<'a> {
    steps: &'a Steps,
    placed: Vec<Placed>,
    spans: Vec<(f64, f64)>,
}

fn place_chain<'a>(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &Camera,
    steps: &'a Steps,
) -> Result<Chain<'a>> {
    let Steps {
        nodes,
        entries,
        edges,
        ln_masses,
        xs,
    } = steps;
    let depth = cam.anchor.path.len();
    if entries.len() < depth {
        return Err(format!(
            "node {:#018x} does not place branch {} under measure {}",
            nodes[entries.len()],
            cam.anchor.path[entries.len()],
            lens.p
        ));
    }
    let point = Point::new(&cam.anchor.digits);
    let ln_scale = ln_masses[depth] + cam.x;
    let y = |edge: &Point| {
        let (under, ln_mass) = point.to(edge);
        let far = (ln_scale + ln_mass).exp();
        if under {
            cam.a + far
        } else {
            cam.a - far
        }
    };
    let fixed = QLayout::supports(ln_scale) && tree.client_measure(nodes[depth], lens.p).is_some();
    let (anchor, mut span) = if fixed {
        // The camera already places an exact point. In the fixed range
        // it can anchor the whole node without locating a child first.
        let anchor = Placed::new(
            (point.clone(), cam.a),
            (point, cam.a),
            ln_masses[depth],
            ln_scale,
            xs[depth],
        );
        let geometry = anchor.quantile().expect("the camera is in the fixed range");
        (anchor, (geometry.y(0), geometry.y(CDF_SIZE)))
    } else {
        let (top, bottom) = around(tree.distribution(nodes[depth], lens.p)?, &point);
        let (y_top, y_bottom) = (y(&top), y(&bottom));
        let anchor = Placed::new(
            (top, y_top),
            (bottom, y_bottom),
            ln_masses[depth],
            ln_scale,
            xs[depth],
        );
        (anchor, (y(&Point::top()), y(&Point::bottom())))
    };
    let mut placed = Vec::with_capacity(nodes.len());
    let mut spans = Vec::with_capacity(nodes.len());
    placed.push(anchor);
    spans.push(span);
    for (i, (top, bottom)) in edges.iter().enumerate().rev() {
        let own = Placed::new(
            (top.clone(), span.0),
            (bottom.clone(), span.1),
            ln_masses[i],
            ln_masses[i] + cam.x,
            xs[i],
        );
        span = own.span();
        placed.push(own);
        spans.push(span);
    }
    placed.reverse();
    spans.reverse();
    Ok(Chain {
        steps,
        placed,
        spans,
    })
}

struct Layout<'a, T> {
    tree: &'a T,
    lens: Lens,
    cam: &'a Camera,
    right: f64,
    /// A physical row, in view heights: a child shorter than one is drawn
    /// in the rows it falls in.
    row: f64,
    frame: Frame,
}

impl<T: LayoutTree> Layout<'_, T> {
    fn push(&mut self, b: LayoutBox) {
        if b.x1 > self.cam.x && b.x0 < self.right && b.y1 > 0.0 && b.y0 < 1.0 {
            self.frame.boxes.push(b);
        }
    }

    /// Lay out the node's children in view, all but the chain child at
    /// `skip`: their outline and inside as spans of physical rows, and each
    /// item a row or more tall as a box of its own, for its label, picking
    /// and its own children.
    fn lay(&mut self, node: NodeId, placed: Placed, skip: Option<Rank>) -> Result<()> {
        let tree = self.tree;
        self.frame.placed.insert(node, placed.clone());
        let Some(d) = tree.distribution(node, self.lens.p)? else {
            return Ok(());
        };
        if let Some((d, geometry)) = tree
            .client_measure(node, self.lens.p)
            .and_then(|d| Some((d, placed.quantile()?)))
        {
            return match d.fixed() {
                Some(fixed) => self.lay_quantile(node, &placed, skip, d, fixed, &geometry),
                None => self.lay_exact(node, &placed, skip, d, Some(&geometry)),
            };
        }
        self.lay_exact(node, &placed, skip, d, None)
    }

    fn lay_exact(
        &mut self,
        node: NodeId,
        placed: &Placed,
        skip: Option<Rank>,
        d: &impl RankView,
        quantile: Option<&QLayout>,
    ) -> Result<()> {
        let mut heights = Heights::new(placed, d, quantile);
        let mut raster = Raster::new(self.row);
        // Start a row above the view so the first visible step has a width.
        let mut item = d.locate(&placed.at(-self.row));
        let mut visible = std::iter::from_fn(move || {
            let found = item?;
            item = d.next(found);
            let ys = heights.item(found);
            (ys.0 < 1.0).then_some((found, ys))
        });
        if let Some(q) = self.tree.packed_q(node, self.lens) {
            // Packed q does not have a persistent branch index. Batch the
            // visible lookups into one pass through q's packed positions.
            let items: Vec<_> = visible.collect();
            let branches: Vec<_> = items
                .iter()
                .filter_map(|(item, _)| match *item {
                    Item::Child { run, at } => Some(d.run(run).branch(at)),
                    Item::Gap { .. } => None,
                })
                .collect();
            let mut widths = q.ranks(&branches).into_iter().map(|rank| {
                rank.and_then(|rank| q.locate_rank(rank))
                    .map_or(f64::INFINITY, |(run, at)| -q.run(run).ln_p(at))
            });
            for (item, ys) in items {
                let width = matches!(item, Item::Child { .. }).then(|| widths.next().unwrap());
                self.lay_item(node, placed, skip, d, item, ys, width, &mut raster)?;
            }
        } else {
            for (item, ys) in &mut visible {
                self.lay_item(node, placed, skip, d, item, ys, None, &mut raster)?;
            }
        }
        let last = (1.0 / self.row).ceil() as i64;
        for mut span in raster.finish() {
            span.rows.start = span.rows.start.max(0);
            span.rows.end = span.rows.end.min(last);
            if span.rows.start < span.rows.end {
                self.span(node, placed, span);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lay_item(
        &mut self,
        node: NodeId,
        placed: &Placed,
        skip: Option<Rank>,
        d: &impl RankView,
        item: Item,
        (y0, y1): (f64, f64),
        known_width: Option<f64>,
        raster: &mut Raster,
    ) -> Result<()> {
        let width = match item {
            Item::Child { run, at } => {
                let r = d.run(run);
                known_width.map_or_else(
                    || self.tree.nats_q(node, self.lens, r.branch(at), r.ln_p(at)),
                    Ok,
                )?
            }
            Item::Gap { .. } => f64::INFINITY,
        };
        raster.add(y0, y1, width);
        let chain = matches!(item, Item::Child { .. }) && skip == Some(d.item_rank(item));
        if chain || y1 - y0 < self.row {
            return Ok(());
        }
        let b = item_box(
            self.tree,
            self.lens,
            node,
            placed,
            d,
            item,
            (y0, y1),
            matches!(item, Item::Child { .. }).then_some(width),
        )?;
        match item {
            Item::Child { .. } => self.child(node, placed, b, || {
                let (top, bottom) = d.item_edges(item);
                Point::ln_between(&top, &bottom)
            })?,
            Item::Gap { .. } => self.push(b),
        }
        Ok(())
    }

    fn lay_quantile(
        &mut self,
        node: NodeId,
        placed: &Placed,
        skip: Option<Rank>,
        d: &ClientMeasure,
        fixed: &FixedIndex,
        geometry: &QLayout,
    ) -> Result<()> {
        let rows = 1.0 / self.row;
        let row_geometry = geometry.in_rows(self.row);
        let visible =
            row_geometry.position(&fixed.p, -1.0)..row_geometry.tops_before(&fixed.p, rows, 0);
        if visible.is_empty() {
            return Ok(());
        }
        let q = self.tree.client_measure(node, self.lens.q);
        let key = (self.lens.q, q.map_or(0, ClientMeasure::revision));
        let mut cached = fixed.widths.borrow_mut();
        if cached.as_ref().is_none_or(|(prior, _)| *prior != key) {
            *cached = Some((key, d.fixed_widths(fixed, q, self.lens.p == self.lens.q)));
        }
        let (_, widths) = cached.as_ref().expect("a width index for this lens");
        let mut hits = self.frame.hit_pool.pop().unwrap_or_default();
        geometry.tall_in_order(&fixed.p, [-self.row, 1.0], self.row, &mut hits);
        let mut lay_err = Ok(());
        for &hit in &hits {
            let item = fixed.items[hit.branch as usize];
            if skip
                .is_some_and(|rank| matches!(item, Item::Child { .. }) && rank == d.item_rank(item))
            {
                continue;
            }
            let end = u128::from(hit.end) + 1;
            let (y0, y1) = (geometry.y(u128::from(hit.start)), geometry.y(end));
            if y1 - y0 < self.row || y1 <= -self.row || 1.0 <= y0 {
                continue;
            }
            let ln_extent = || cdf_ln(u128::from(hit.start), end);
            let ln_gap = matches!(item, Item::Gap { .. }).then(ln_extent);
            let b = match item_box(
                self.tree,
                self.lens,
                node,
                placed,
                d,
                item,
                (y0, y1),
                ln_gap,
            ) {
                Ok(b) => b,
                Err(err) => {
                    lay_err = Err(err);
                    break;
                }
            };
            let res = match item {
                Item::Child { .. } => self.child(node, placed, b, ln_extent),
                Item::Gap { .. } => {
                    self.push(b);
                    Ok(())
                }
            };
            if let Err(err) = res {
                lay_err = Err(err);
                break;
            }
        }
        hits.clear();
        self.frame.hit_pool.push(hits);
        lay_err?;

        quantile_rows(&row_geometry, fixed, widths, rows, visible, |span| {
            self.span(node, placed, span)
        });
        Ok(())
    }

    /// A span of a node's rows, where it shows.
    fn span(&mut self, node: NodeId, placed: &Placed, span: RowSpan) {
        let reach = placed.x + span.reach;
        if reach <= self.cam.x || self.right <= placed.x {
            return;
        }
        let (y0, y1) = (
            span.rows.start as f64 * self.row,
            span.rows.end as f64 * self.row,
        );
        self.frame.spans.push(Span {
            node,
            y0,
            y1,
            x0: placed.x,
            fill: if span.boundary { placed.x } else { reach },
            reach,
        });
    }

    /// A child's box, and its children where they start in view.
    fn child(
        &mut self,
        node: NodeId,
        placed: &Placed,
        b: LayoutBox,
        ln_extent: impl FnOnce() -> f64,
    ) -> Result<()> {
        let (x1, kind, ys, branch) = (b.x1, b.kind, (b.y0, b.y1), b.branch);
        self.push(b);
        if self.right <= x1 || kind != BoxKind::Node {
            return Ok(());
        }
        let id = child_hash(node, branch.expect("an expanded node has a branch"));
        self.lay(id, placed.child(ys, ln_extent(), x1), None)
    }
}

/// Consecutive physical rows with the same right edge and boundary coverage.
#[derive(Clone, Debug, PartialEq)]
struct RowSpan {
    rows: Range<i64>,
    reach: f64,
    boundary: bool,
}

impl RowSpan {
    fn join(&mut self, next: &Self) -> bool {
        if self.rows.end != next.rows.start
            || self.reach != next.reach
            || self.boundary != next.boundary
        {
            return false;
        }
        self.rows.end = next.rows.end;
        true
    }
}

/// A node's children as physical rows. The pixel at an edge of any child
/// is outline, including shared horizontal edges. An edge on a grid line
/// belongs to the pixel below it or to its right.
struct Raster {
    /// A physical row, in view heights.
    height: f64,
    /// The width of the item before, none for a hole, and where it ends,
    /// in rows; none before the first.
    before: Option<(Option<f64>, f64)>,
    open: Option<RowSpan>,
    spans: Vec<RowSpan>,
}

impl Raster {
    fn new(height: f64) -> Self {
        Self {
            height,
            before: None,
            open: None,
            spans: Vec::new(),
        }
    }

    /// The row a coordinate in rows falls in, a grid line the one below it.
    fn row(at: f64) -> i64 {
        (at + ON_GRID).floor() as i64
    }

    /// An item from `y0` to `y1` view heights, drawn `width` wide; a width
    /// that is not finite is a hole. Items come in order down the node,
    /// from one above the view: the first is where the children start.
    fn add(&mut self, y0: f64, y1: f64, width: f64) {
        // Rows beyond the one past the view on either side show nothing,
        // and a far view's rows would be no whole number.
        let rows = |y: f64| (y / self.height).clamp(-2.0, 2.0 + 1.0 / self.height);
        let (top, bottom) = (rows(y0), rows(y1));
        let (first, last) = (Self::row(top), Self::row(bottom - 2.0 * ON_GRID));
        if last < first {
            return;
        }
        let width = width.is_finite().then_some(width);
        // Both children's horizontal edges start at the node's left edge.
        let before = self.before.and_then(|(width, _)| width).unwrap_or(0.0);
        self.edge(first, before.max(width.unwrap_or(0.0)), true);
        if let Some(width) = width {
            // The rows between, wholly its own: no edge but its end.
            if first + 1 < last {
                self.flush();
                self.emit(RowSpan {
                    rows: first + 1..last,
                    boundary: false,
                    reach: width,
                });
            }
            self.edge(last, width, false);
        }
        self.before = Some((width, bottom));
    }

    /// A side or horizontal boundary of a child in this row.
    fn edge(&mut self, index: i64, width: f64, boundary: bool) {
        if self
            .open
            .as_ref()
            .is_some_and(|row| row.rows.start != index)
        {
            self.flush();
        }
        let row = self.open.get_or_insert(RowSpan {
            rows: index..index + 1,
            reach: width,
            boundary,
        });
        row.reach = row.reach.max(width);
        row.boundary |= boundary;
    }

    fn flush(&mut self) {
        if let Some(row) = self.open.take() {
            self.emit(row);
        }
    }

    /// A span, joined to the one before where it continues it alike.
    fn emit(&mut self, span: RowSpan) {
        if !self.spans.last_mut().is_some_and(|last| last.join(&span)) {
            self.spans.push(span);
        }
    }

    /// The step at the bottom of the last item, to where the children start.
    fn finish(mut self) -> Vec<RowSpan> {
        if let Some((width, bottom)) = self.before {
            let row = Self::row(bottom);
            self.edge(row, width.unwrap_or(0.0), true);
        }
        self.flush();
        self.spans
    }
}

fn cdf_ln(start: u128, bottom: u128) -> f64 {
    ((bottom - start) as f64 / CDF_SIZE as f64).ln()
}

/// Every horizontal boundary in a row joins its adjacent children's widths.
/// At a grid line the preceding child still contributes its bottom edge.
fn quantile_rows(
    geometry: &QLayout,
    fixed: &FixedIndex,
    widths: &WidthIndex,
    rows: f64,
    visible: Range<usize>,
    mut out: impl FnMut(RowSpan),
) {
    let p = &fixed.p;
    let limit = visible.end;
    let mut cursor = visible.start;
    let mut before = None;
    let mut open: Option<RowSpan> = None;
    let mut emit = |next: RowSpan| {
        if open.as_mut().is_some_and(|prior| prior.join(&next)) {
            return;
        }
        if let Some(prior) = open.replace(next) {
            out(prior);
        }
    };
    let mut index = Raster::row(geometry.y(0)).max(-1);
    let end_row = Raster::row(geometry.y(CDF_SIZE))
        .saturating_add(1)
        .min(rows.ceil() as i64);
    let mut last_end = f64::NEG_INFINITY;
    while index < end_row {
        let lower = index as f64 + ON_GRID;
        let first = if lower <= last_end {
            cursor
        } else {
            geometry.bottom_at(p, lower, cursor)
        };
        let end = geometry
            .tops_before(p, index as f64 + 1.0 - ON_GRID, first)
            .min(limit);
        let mut reach = 0.0;
        let mut boundary = false;
        if first < end {
            reach = widths.maximum(first, end);
            boundary = first + 1 < end;
            if before != Some(first) {
                let top = geometry.y(QLayout::start(p, first));
                if index as f64 <= top + ON_GRID {
                    boundary = true;
                    reach = reach.max(before.map_or(0.0, |at| widths.at(at)));
                }
            }
            before = Some(end - 1);
            cursor = end - 1;
            last_end = geometry.end_y(p.ends()[cursor]);
        }
        if before.is_some_and(|at| at + 1 == limit) && index == Raster::row(last_end) {
            boundary = true;
            reach = reach.max(widths.at(cursor));
        }
        if 0 <= index && 0.0 < reach {
            emit(RowSpan {
                rows: index..index + 1,
                boundary,
                reach,
            });
        }
        index += 1;
        // Skip the whole interior of a tall child, leaving its final row
        // for the next boundary. Most rows never need this rounding.
        if first < end && (index + 1) as f64 <= last_end - ON_GRID {
            let last = Raster::row(last_end - 2.0 * ON_GRID).min(end_row);
            let width = widths.at(cursor);
            if index < last && 0.0 < width {
                emit(RowSpan {
                    rows: index.max(0)..last,
                    boundary: false,
                    reach: width,
                });
            }
            index = last;
        }
    }
    if let Some(prior) = open {
        out(prior);
    }
}

pub fn layout(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    row_px: f64,
) -> Result<Frame> {
    layout_into(tree, lens, cam, view, row_px, Frame::default())
}

/// Lay out into a previous frame's storage.
pub fn layout_into(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    row_px: f64,
    frame: Frame,
) -> Result<Frame> {
    let steps = steps(tree, lens, cam.origin, &cam.anchor.path)?;
    let chain = place_chain(tree, lens, cam, &steps)?;
    layout_chain(tree, lens, cam, view, row_px, frame, chain)
}

/// Settle and clamp a camera, then draw and describe the same projection.
pub fn project(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &mut Camera,
    viewport: Viewport,
    pixel_ratio: f64,
    frame: Frame,
) -> Result<(Frame, View)> {
    let (anchor, steps) = settled(tree, lens, cam)?;
    cam.anchor = anchor;
    let mut chain = place_chain(tree, lens, cam, &steps)?;
    if cam.origin == ROOT {
        let a = clamp_offset(cam.a, chain.spans[0]);
        if a != cam.a {
            cam.a = a;
            // Place from the new offset to preserve rounding at pixel edges.
            chain = place_chain(tree, lens, cam, &steps)?;
        }
    }
    let demand = chain.view(cam, viewport, viewport.height * pixel_ratio);
    let frame = layout_chain(tree, lens, cam, viewport, 1.0 / pixel_ratio, frame, chain)?;
    Ok((frame, demand))
}

fn layout_chain(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    row_px: f64,
    mut frame: Frame,
    chain: Chain<'_>,
) -> Result<Frame> {
    frame.clear();
    let Chain {
        steps: Steps { nodes, entries, .. },
        placed,
        spans,
    } = chain;
    (frame.origin_top, frame.origin_bottom) = spans[0];
    let mut context = Layout {
        tree,
        lens,
        cam,
        right: cam.x + view.width / cam.k,
        row: row_px / view.height,
        frame,
    };
    let depth = nodes.len() - 1;
    context.frame.top = nodes[depth];
    context.lay(nodes[depth], placed[depth].clone(), None)?;
    for i in (0..depth).rev() {
        let (own, child, entry) = (&placed[i], &placed[i + 1], &entries[i]);
        let (y0, y1) = spans[i + 1];
        context.push(LayoutBox {
            node: nodes[i],
            rank: entry.rank,
            count: 1,
            kind: BoxKind::Node,
            branch: Some(entry.branch),
            p: entry.ln_p.exp(),
            q: if entry.nats_q.is_finite() {
                (-entry.nats_q).exp()
            } else {
                f64::NAN
            },
            ln_mass: child.ln_mass,
            x0: own.x,
            x1: child.x,
            y0,
            y1,
        });
        context.frame.top = nodes[i];
        context.lay(nodes[i], own.clone(), Some(entry.rank))?;
        // Covering the view with its own box left of it, this node hides
        // every remaining ancestor's other children.
        let (top, bottom) = spans[i];
        if top <= 0.0 && bottom >= 1.0 && own.x <= cam.x {
            break;
        }
    }
    Ok(context.frame)
}

/// The known children of a node the frame laid out, those that reach into
/// the view, top to bottom: each one's branch and its edges in view heights,
/// however thin, even where the frame draws it in a row with others.
pub fn known(
    tree: &impl LayoutTree,
    lens: Lens,
    frame: &Frame,
    node: NodeId,
) -> Result<Vec<(Branch, f64, f64)>> {
    let (Some(placed), Some(d)) = (frame.placed.get(&node), tree.distribution(node, lens.p)?)
    else {
        return Ok(Vec::new());
    };
    let quantile = tree
        .client_measure(node, lens.p)
        .and_then(|d| Some((d, placed.quantile()?)));
    if let Some((d, geometry, fixed)) = quantile
        .as_ref()
        .and_then(|(d, geometry)| Some((*d, geometry, d.fixed()?)))
    {
        let first = geometry.position(&fixed.p, 0.0);
        let end = geometry.tops_before(&fixed.p, 1.0, first);
        let known = (first..end)
            .filter_map(|position| {
                let Item::Child { run, at } = fixed.items[position] else {
                    return None;
                };
                let (y0, y1) = geometry.span(&fixed.p, position);
                Some((d.run(run).branch(at), y0, y1))
            })
            .collect();
        return Ok(known);
    }
    let mut heights = Heights::new(placed, d, quantile.as_ref().map(|(_, geometry)| geometry));
    let mut known = Vec::new();
    let mut item = d.locate(&placed.at(0.0));
    while let Some(found) = item {
        let (y0, y1) = heights.item(found);
        if 1.0 <= y0 {
            break;
        }
        if let Item::Child { run, at } = found {
            known.push((d.run(run).branch(at), y0, y1));
        }
        item = d.next(found);
    }
    Ok(known)
}

/// What the camera looks at, as a session's view of `rows` pixel rows: the
/// deepest node on the camera's path that holds the whole view, the view's
/// top in it, exact as a frame places it, ln of the view's height in it, and
/// how far right the screen reaches past where its children start. Where the
/// view reaches past the origin, what of it the origin holds.
pub fn view(
    tree: &impl LayoutTree,
    lens: Lens,
    cam: &Camera,
    viewport: Viewport,
    rows: f64,
) -> Result<View> {
    let steps = steps(tree, lens, cam.origin, &cam.anchor.path)?;
    Ok(place_chain(tree, lens, cam, &steps)?.view(cam, viewport, rows))
}

impl Chain<'_> {
    fn view(&self, cam: &Camera, viewport: Viewport, rows: f64) -> View {
        let Chain {
            steps: Steps { nodes, .. },
            placed,
            spans,
        } = self;
        let right = cam.x + viewport.width / cam.k;
        let covers = |&(top, bottom): &(f64, f64)| top <= 0.0 && 1.0 <= bottom;
        // A node is placed only where q is known above it: the view is told
        // from the deepest one the client can place, so the source tells what
        // places the rest, though p's mass may all go one way.
        let known = placed
            .iter()
            .position(|at| !at.x.is_finite())
            .unwrap_or(placed.len());
        let (depth, top, ln_height, rows) = match spans[..known].iter().rposition(covers) {
            Some(depth) => {
                let at = &placed[depth];
                let point = (depth == nodes.len() - 1).then(|| Point::new(&cam.anchor.digits));
                let top = view_top(at, point.as_ref(), cam.a).unwrap_or_else(|| Point::new(&[]));
                (
                    depth,
                    top.floor(precision(at.ln_scale)),
                    0.0 - at.ln_scale,
                    rows,
                )
            }
            None => {
                // The view is at least as tall as the part of the origin it
                // shows: floats of the origin's mass tell that part.
                let (top, bottom) = spans[0];
                let (from, to) = (top.max(0.0), bottom.min(1.0));
                let at = &placed[0];
                let start = Point::top()
                    .moved(true, from - top, -at.ln_scale)
                    .unwrap_or_else(Point::top);
                let shown = (to - from).max(0.0);
                (0, start, shown.ln() - at.ln_scale, rows * shown)
            }
        };
        View {
            node: nodes[depth],
            top: top.digits(),
            ln_height,
            rows,
            right: right - placed[depth].x,
        }
    }
}

/// The top of the view as a point of a node holding it, told from whichever
/// is nearest of what is exact there: the node's top, its bottom and, in the
/// camera's own node, the camera's `point`, `a` view heights below the top.
fn view_top(placed: &Placed, point: Option<&Point>, a: f64) -> Option<Point> {
    let (y0, y1) = placed.span();
    if !placed.ln_scale.is_finite() {
        return None;
    }
    match point {
        Some(point) if a.abs() <= y0.abs().min(y1.abs()) => {
            point.moved(a <= 0.0, a.abs(), -placed.ln_scale)
        }
        _ if y1.abs() < y0.abs() => Point::bottom().moved(y1 <= 0.0, y1.abs(), -placed.ln_scale),
        _ => Point::top().moved(true, y0.abs(), -placed.ln_scale),
    }
}

/// Where the node `path` leads to from an origin sits, as layout places it:
/// ln of its p mass below the origin, and where its children start, in nats
/// right of the origin's. As far as p places the path: `placed` of its
/// branches.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Reach {
    pub ln_mass: f64,
    pub x: f64,
    pub placed: usize,
}

pub fn reach(tree: &impl LayoutTree, lens: Lens, origin: NodeId, path: &[Branch]) -> Result<Reach> {
    let steps = steps(tree, lens, origin, path)?;
    let placed = steps.entries.len();
    Ok(Reach {
        ln_mass: steps.ln_masses[placed],
        x: steps.xs[placed],
        placed,
    })
}

/// The edges in view heights of the node `depth` steps down the camera's
/// path from its origin, as far as p places that path.
pub fn extent(tree: &impl LayoutTree, lens: Lens, cam: &Camera, depth: usize) -> Result<Extent> {
    let steps = steps(tree, lens, cam.origin, &cam.anchor.path)?;
    let spans = place_chain(tree, lens, cam, &steps)?.spans;
    let &(top, bottom) = spans.get(depth).ok_or_else(|| {
        format!(
            "the camera's path is placed {} deep, not {depth}",
            spans.len() - 1
        )
    })?;
    Ok(Extent { top, bottom })
}

/// Keep the view inside the root's interval, returning only the new anchor offset.
pub fn clamp_position(tree: &impl LayoutTree, lens: Lens, cam: &Camera) -> Result<f64> {
    // Only the root's interval is all there is; beyond any other origin lies
    // its parent, unknown until fetched.
    if cam.origin != crate::tree::ROOT {
        return Ok(cam.a);
    }
    let Extent { top, bottom } = extent(tree, lens, cam, 0)?;
    Ok(clamp_offset(cam.a, (top, bottom)))
}

fn clamp_offset(a: f64, (top, bottom): (f64, f64)) -> f64 {
    if 0.0 < top {
        a - top
    } else if bottom < 1.0 {
        a + 1.0 - bottom
    } else {
        a
    }
}

/// The camera's point under the name that keeps the view exact: its path
/// down to the deepest node that both holds the point and covers the view,
/// its digits within that node. Nothing on screen moves.
///
/// The name follows from the view and the tree alone. Going up a node is
/// exact and keeps every digit, so a view that zooms out and back in on a
/// point it has not re-aimed returns to the very same anchor. Where p no
/// longer places a branch of the path, as after a lens or a replaced
/// distribution, the path ends there, at that node's top.
pub fn settle(tree: &impl LayoutTree, lens: Lens, cam: &Camera) -> Result<Anchor> {
    settled(tree, lens, cam).map(|(anchor, _)| anchor)
}

fn settled(tree: &impl LayoutTree, lens: Lens, cam: &Camera) -> Result<(Anchor, Steps)> {
    let mut steps = steps(tree, lens, cam.origin, &cam.anchor.path)?;
    let mut depth = steps.entries.len();
    let mut point = if depth == cam.anchor.path.len() {
        Point::new(&cam.anchor.digits)
    } else {
        Point::new(&[])
    };
    // Whether the part of a node between two of its edges covers the view,
    // the node e^ln_scale view heights tall. A node that fits the view
    // covers it, whichever way its edges round.
    let covers = |point: &Point, ln_scale: f64, top: &Point, bottom: &Point| {
        let ((_, above), (_, under)) = (point.to(top), point.to(bottom));
        cam.a - (ln_scale + above).exp() <= 0.0 && 1.0 <= cam.a + (ln_scale + under).exp()
    };
    while let Some((top, bottom)) = depth.checked_sub(1).map(|i| &steps.edges[i]) {
        let ln_scale = steps.ln_masses[depth] + cam.x;
        if covers(&point, ln_scale, &Point::top(), &Point::bottom()) {
            break;
        }
        point = point.out_of_child(top, bottom);
        depth -= 1;
    }
    steps.truncate(depth);
    let mut ln_scale = steps.ln_masses[depth] + cam.x;
    while let Some(d) = tree.distribution(steps.nodes[depth], lens.p)? {
        let Some(item @ Item::Child { run, at }) = d.locate(&point) else {
            break;
        };
        let (top, bottom) = d.item_edges(item);
        if !point.within(&top, &bottom) || !covers(&point, ln_scale, &top, &bottom) {
            break;
        }
        ln_scale += steps.push(tree, lens, d, (run, at), (top, bottom))?;
        let (top, bottom) = &steps.edges[depth];
        point = point.into_child(top, bottom, precision(ln_scale));
        depth += 1;
    }
    let anchor = Anchor {
        path: steps.entries.iter().map(|entry| entry.branch).collect(),
        digits: point.digits(),
    };
    Ok((anchor, steps))
}

/// The camera's point moved to what the view shows `fy` view heights down,
/// to 2^-64 of a view height, under its settled name: the point a zoom about
/// `fy` then leaves where it is, so only depth changes. Unchanged where the
/// point is there already, or where the origin does not reach that far.
pub fn aim(tree: &impl LayoutTree, lens: Lens, cam: &Camera, fy: f64) -> Result<Aim> {
    let same = Aim {
        anchor: cam.anchor.clone(),
        a: cam.a,
    };
    let by = fy - cam.a;
    if by == 0.0 || !by.is_finite() {
        return Ok(same);
    }
    let (Anchor { path, digits }, steps) = settled(tree, lens, cam)?;
    let ln_scale = steps.ln_masses[path.len()] + cam.x;
    if !ln_scale.is_finite() {
        return Ok(same);
    }
    let moved = Point::new(&digits).moved(0.0 < by, by.abs(), -ln_scale);
    Ok(moved.map_or(same, |point| Aim {
        anchor: Anchor {
            path,
            digits: point.floor(precision(ln_scale)).digits(),
        },
        a: fy,
    }))
}

/// The box a pixel shows, `pixel_px` square with its top-left corner at
/// `(x_px, y_px)`: a child, however thin, found by walking down
/// from the origin through the nodes the frame laid out.
#[allow(clippy::too_many_arguments)]
pub fn pick(
    tree: &impl LayoutTree,
    lens: Lens,
    frame: &Frame,
    cam: &Camera,
    view: Viewport,
    x_px: f64,
    y_px: f64,
    pixel_px: f64,
) -> Result<Option<PickResult>> {
    if !(0.0..view.width).contains(&x_px) || !(0.0..view.height).contains(&y_px) {
        return Ok(None);
    }
    // The pixel's corner, as drawing has it: an edge within float error past
    // a grid line is on it, so the corner is past the edge too.
    let (x, y) = (
        cam.x + x_px / cam.k,
        (y_px + ON_GRID * pixel_px) / view.height,
    );
    // Outside every known child, the origin itself.
    let origin = PickResult {
        r#box: LayoutBox {
            node: cam.origin,
            rank: 0,
            count: 0,
            kind: BoxKind::Node,
            branch: None,
            p: 1.0,
            q: 1.0,
            ln_mass: 0.0,
            x0: 0.0,
            x1: 0.0,
            y0: frame.origin_top.min(0.0),
            y1: frame.origin_bottom.max(1.0),
        },
        exact: false,
        ancestors: Vec::new(),
    };
    if !(frame.origin_top..frame.origin_bottom).contains(&y) || x < 0.0 {
        return Ok(Some(origin));
    }
    let mut node = frame.top;
    // The box of the node the pointer is in, below the origin.
    let mut within = None;
    let mut ancestors = Vec::new();
    loop {
        let (Some(placed), Some(d)) = (frame.placed.get(&node), tree.distribution(node, lens.p)?)
        else {
            return Ok(Some(origin));
        };
        // Nothing at any height: a node known to have no children, whose
        // box is all there is.
        let none = || {
            Some(within.map_or_else(
                || origin.clone(),
                |b| PickResult {
                    r#box: b,
                    exact: true,
                    ancestors: ancestors.clone(),
                },
            ))
        };
        let pixel_bottom = y + pixel_px / view.height;
        let quantile = tree
            .client_measure(node, lens.p)
            .and_then(|d| Some((d, placed.quantile()?)));
        let b = if let Some((d, geometry, fixed)) = quantile
            .as_ref()
            .and_then(|(d, geometry)| Some((*d, geometry, d.fixed()?)))
        {
            let mut position = geometry.position(&fixed.p, y);
            let Some(&item) = fixed.items.get(position) else {
                return Ok(none());
            };
            let mut edges = geometry.span(&fixed.p, position);
            if matches!(item, Item::Gap { .. })
                && edges.1 - edges.0 < pixel_bottom - y
                && edges.1 < pixel_bottom
                && matches!(fixed.items.get(position + 1), Some(Item::Child { .. }))
            {
                position += 1;
                edges = geometry.span(&fixed.p, position);
            }
            let item = fixed.items[position];
            let ln_gap = matches!(item, Item::Gap { .. }).then(|| {
                cdf_ln(
                    QLayout::start(&fixed.p, position),
                    u128::from(fixed.p.ends()[position]) + 1,
                )
            });
            item_box(tree, lens, node, placed, d, item, edges, ln_gap)?
        } else {
            let Some(item) = d.locate(&placed.at(y)) else {
                return Ok(none());
            };
            let mut heights =
                Heights::new(placed, d, quantile.as_ref().map(|(_, geometry)| geometry));
            let item = pixel_child(&mut heights, item, y, pixel_bottom);
            item_box(tree, lens, node, placed, d, item, heights.item(item), None)?
        };
        let Some(branch) = b.branch.filter(|_| b.x1 <= x) else {
            let exact = b.branch.is_some();
            return Ok(Some(PickResult {
                r#box: b,
                exact,
                ancestors,
            }));
        };
        let child = child_hash(node, branch);
        if !frame.placed.contains_key(&child) || tree.distribution(child, lens.p)?.is_none() {
            return Ok(Some(PickResult {
                r#box: b,
                exact: false,
                ancestors,
            }));
        }
        ancestors.push(child);
        within = Some(b);
        node = child;
    }
}

/// The item a pixel from `y` down to `bottom` selects, its corner in
/// `item`: a gap thinner than the pixel is drawn in the row it falls in,
/// with the children there, so the known child after it in the pixel is
/// selected instead. A gap a pixel or more tall is drawn as its own box and
/// keeps its pixels.
fn pixel_child(heights: &mut Heights<'_, impl RankView>, item: Item, y: f64, bottom: f64) -> Item {
    if !matches!(item, Item::Gap { .. }) {
        return item;
    }
    let (top, end) = heights.item(item);
    if bottom - y <= end - top {
        return item;
    }
    match heights.d.next(item) {
        Some(next @ Item::Child { .. }) if heights.item(next).0 < bottom => next,
        _ => item,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Run;
    use crate::tree::ROOT;
    use std::f64::consts::LN_2;

    #[test]
    fn clipped_windows_reach_children_to_the_right_of_hidden_parents() {
        let mut tree = ClientTree::new();
        let equal = || Run::new(0, vec![0, 1], vec![-LN_2; 2], &[]).unwrap();
        tree.insert(ROOT, 0, vec![equal()], Some(2)).unwrap();
        let child = tree.child(ROOT, 0).unwrap();
        tree.insert(child, 0, vec![equal()], Some(2)).unwrap();
        let cam = Camera {
            origin: ROOT,
            x: 1.0,
            k: 1.0,
            anchor: Anchor::default(),
            a: 0.0,
        };
        let placed = Placed::new(
            (Point::new(&[0x40]), 0.0),
            (Point::new(&[0xc0]), 1.0),
            0.0,
            LN_2,
            0.0,
        );
        let mut layout = Layout {
            tree: &tree,
            lens: LENS,
            cam: &cam,
            right: 4.0,
            row: 0.25,
            frame: Frame::default(),
        };
        layout.lay(ROOT, placed, None).unwrap();
        assert_eq!(layout.frame.boxes.len(), 1);
        let b = &layout.frame.boxes[0];
        assert_eq!((b.node, b.branch, b.y0, b.y1), (child, Some(1), 0.0, 0.5));
        // The visible child has one filled row between its boundary rows.
        let spans = &layout.frame.spans;
        assert_eq!(spans.len(), 3);
        assert_eq!(
            (spans[0].y0, spans[0].y1, spans[0].fill, spans[0].reach),
            (0.0, 0.25, b.x0, b.x1)
        );
        assert_eq!((spans[1].y0, spans[1].y1, spans[1].fill), (0.25, 0.5, b.x1));
        assert_eq!(
            (spans[2].y0, spans[2].y1, spans[2].fill, spans[2].reach),
            (0.5, 0.75, b.x0, b.x1)
        );
    }

    #[test]
    fn a_missing_q_stays_unknown_and_sparse_q_uses_branch_ids() {
        let mut tree = ClientTree::new();
        let equal = Run::new(0, vec![0, 1], vec![-LN_2; 2], &[]).unwrap();
        tree.insert(ROOT, 0, vec![equal], Some(2)).unwrap();
        let cam = Camera {
            origin: ROOT,
            x: 0.0,
            k: 1.0,
            anchor: Anchor::default(),
            a: 0.0,
        };
        let view = Viewport {
            width: 4.0,
            height: 2.0,
        };
        let lens = Lens { p: 0, q: 1 };
        let frame = layout(&tree, lens, &cam, view, 1.0).unwrap();
        assert_eq!(frame.boxes.len(), 2);
        assert!(frame.boxes.iter().all(|b| b.x1 == f64::INFINITY));
        let q = Run::new(0, vec![1, 0], vec![-3.0, -2.0], &[]).unwrap();
        tree.insert(ROOT, 1, vec![q], None).unwrap();
        let frame = layout(&tree, lens, &cam, view, 1.0).unwrap();
        assert_eq!((frame.boxes[0].x1, frame.boxes[1].x1), (2.0, 3.0));

        let (_, _, branches, weights) = crate::pstore::normalize_logits(&[f32::MAX; 2]);
        let normalized = Run::new(
            0,
            branches,
            weights.into_iter().map(f64::from).collect(),
            &[],
        )
        .unwrap();
        tree.insert(ROOT, 2, vec![normalized], Some(2)).unwrap();
        assert_eq!(
            tree.nats_q(ROOT, Lens { p: 0, q: 2 }, 0, 0.0) as f32,
            std::f32::consts::LN_2,
        );
    }

    #[test]
    fn fixed_layout_draws_the_same_sibling_steps_as_exact_layout() {
        let p = Run::new(
            0,
            vec![3, 5, 8, 9],
            [0.5f64, 0.3, 0.15, 0.05].into_iter().map(f64::ln).collect(),
            &[],
        )
        .unwrap();
        let q = Run::new(0, vec![9, 3, 8, 5], vec![-1.0, -2.5, -2.5, -3.0], &[]).unwrap();
        let mut exact = Tree::new();
        let mut quantile = ClientTree::new();
        for (measure, run) in [p, q].into_iter().enumerate() {
            exact
                .insert(ROOT, measure, vec![run.clone()], Some(4))
                .unwrap();
            quantile.insert(ROOT, measure, vec![run], Some(4)).unwrap();
        }
        for lens in [LENS, Lens { p: 0, q: 1 }] {
            for dpr in [1.0, 1.25, 1.5, 2.0] {
                for phase in [0.0, 0.2, 0.61, 0.99] {
                    let camera = cam(ROOT, 0.0, top(&[]), phase / VIEW.height);
                    let expected = layout(&exact, lens, &camera, VIEW, 1.0 / dpr).unwrap();
                    let actual = layout(&quantile, lens, &camera, VIEW, 1.0 / dpr).unwrap();
                    let visible = |frame: Frame| {
                        let height = VIEW.height * dpr;
                        let mut rows = vec![None; height.floor() as usize];
                        for span in frame.spans {
                            let first = (span.y0 * height).round() as usize;
                            let last = ((span.y1 * height).round() as usize).min(rows.len());
                            if first < last {
                                rows[first..last].fill(Some((span.fill, span.reach)));
                            }
                        }
                        rows
                    };
                    assert_eq!(
                        visible(actual),
                        visible(expected),
                        "lens {lens:?}, DPR {dpr}, phase {phase}"
                    );
                }
            }
        }
    }

    #[test]
    fn coincident_siblings_keep_the_widest_boundary() {
        let mut tree = ClientTree::new();
        let run = |rank, ln_p, top: &[u8]| Run::new(rank, vec![rank], vec![ln_p], top).unwrap();
        tree.insert(
            ROOT,
            0,
            vec![
                run(0, -LN_2, &[]),
                run(1, 0.1f64.ln(), &[0x80]),
                run(2, -LN_2, &[0x80]),
            ],
            Some(3),
        )
        .unwrap();
        let q = Run::new(0, vec![0, 1, 2], vec![-2.0, -7.0, -3.0], &[]).unwrap();
        tree.insert(ROOT, 1, vec![q], Some(3)).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let camera = cam(ROOT, 0.0, top(&[]), 0.375 / VIEW.height);
        let frame = layout(&tree, lens, &camera, VIEW, 1.0).unwrap();
        let fixed = tree.distribution(ROOT, 0).unwrap().fixed().unwrap();
        assert_eq!(fixed.p.locate(1 << 63).unwrap().branch, 2);
        let row = spans(&frame, ROOT)
            .into_iter()
            .find(|span| span.0 <= 400.0 && 400.0 < span.1)
            .unwrap();
        assert_eq!(row, (400.0, 401.0, 0.0, 7.0));
        let known = known(&tree, lens, &frame, ROOT).unwrap();
        assert_eq!(
            known
                .iter()
                .map(|&(branch, _, _)| branch)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(known[1].1, known[1].2);
    }

    #[test]
    fn overlapping_wire_tops_keep_their_boundaries_in_the_fixed_range() {
        let runs = vec![
            listed(0, &[0], &[0.625], 0.0),
            listed(2, &[2], &[0.390625], 0.609375),
        ];
        let mut client = ClientTree::new();
        let mut exact = Tree::new();
        client.insert(ROOT, 0, runs.clone(), Some(3)).unwrap();
        exact.insert(ROOT, 0, runs, Some(3)).unwrap();
        let camera = cam(ROOT, 0.0, top(&[]), 0.375 / VIEW.height);
        let actual = layout(&client, LENS, &camera, VIEW, 1.0).unwrap();
        let expected = layout(&exact, LENS, &camera, VIEW, 1.0).unwrap();
        assert!(client.distribution(ROOT, 0).unwrap().fixed().is_none());
        near_all(&spans(&actual, ROOT), &spans(&expected, ROOT));
    }

    const VIEW: Viewport = Viewport {
        width: 1000.0,
        height: 800.0,
    };
    const LENS: Lens = Lens { p: 0, q: 0 };

    fn weights(weights: &[f64]) -> Distribution {
        Distribution::exact(weights)
    }

    /// A run of these weights, `above` of the node above it, to 8 digits.
    fn listed(rank: Rank, branches: &[Branch], weights: &[f64], above: f64) -> Run {
        Run::weighed(rank, branches, weights, &Point::top().plus(above), 8)
    }

    fn cam(origin: NodeId, x: f64, anchor: Anchor, a: f64) -> Camera {
        Camera {
            origin,
            x,
            k: 100.0,
            anchor,
            a,
        }
    }

    /// The top of the node `path` leads to.
    fn top(path: &[Branch]) -> Anchor {
        at(path, &[])
    }

    fn at(path: &[Branch], digits: &[u8]) -> Anchor {
        Anchor {
            path: path.to_vec(),
            digits: digits.to_vec(),
        }
    }

    fn near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-11,
            "{actual} differs from {expected}"
        );
    }

    /// Every node along `path` splits by `weights`.
    fn chain_tree(weights_: &[f64], path: &[Branch]) -> (Tree, NodeId) {
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(weights_)).unwrap();
        let mut node = ROOT;
        for &branch in path {
            node = tree.child(node, branch).unwrap();
            tree.set(node, 0, weights(weights_)).unwrap();
        }
        (tree, node)
    }

    #[test]
    fn heights_split_by_p_in_p_order_and_widths_follow_q() {
        let (tree, _) = chain_tree(&[0.25, 0.5, 0.25], &[]);
        // Reaching a child places it as layout draws it, as far as p places
        // the path.
        let at = reach(&tree, LENS, ROOT, &[1, 7]).unwrap();
        assert_eq!(at.placed, 1);
        near(at.x, 2f64.ln());
        near(at.ln_mass, 0.5f64.ln());
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let spans: Vec<_> = frame
            .boxes
            .iter()
            .map(|b| (b.branch, b.y0, b.y1, b.x1 - b.x0))
            .collect();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].0, Some(1));
        near(spans[0].2, 0.5);
        near(spans[0].3, 2f64.ln());
        assert_eq!((spans[1].0, spans[2].0), (Some(0), Some(2)));
        near(spans[2].1, 0.75);
        near(spans[2].2, 1.0);
        near(spans[2].3, 4f64.ln());

        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.25, 0.75])).unwrap();
        tree.set(ROOT, 1, weights(&[0.5, 0.5])).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let frame = layout(&tree, lens, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        // p's order: branch 1 first; q sets both widths alike.
        assert_eq!(frame.boxes[0].branch, Some(1));
        near(frame.boxes[0].y1, 0.75);
        near(frame.boxes[0].x1, 2f64.ln());
        near(frame.boxes[1].x1, 2f64.ln());
    }

    /// A node's spans: rows in pixels, and how far from where its children
    /// start each is filled and outlined.
    fn spans(frame: &Frame, node: NodeId) -> Vec<(f64, f64, f64, f64)> {
        frame
            .spans
            .iter()
            .filter(|span| span.node == node)
            .map(|span| {
                (
                    span.y0 * VIEW.height,
                    span.y1 * VIEW.height,
                    span.fill - span.x0,
                    span.reach - span.x0,
                )
            })
            .collect()
    }

    fn near_all(actual: &[(f64, f64, f64, f64)], expected: &[(f64, f64, f64, f64)]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        for (a, e) in actual.iter().zip(expected) {
            for (x, y) in [(a.0, e.0), (a.1, e.1), (a.2, e.2), (a.3, e.3)] {
                near(x, y);
            }
        }
    }

    #[test]
    fn thin_children_are_filled_and_outlined_where_they_fall_and_pick_apart() {
        // Two thick children, the second ending at 798.8, then three
        // thinner than a pixel: a row holding their horizontal edges
        // is outlined from the left to the widest child.
        let weights_ = [0.5, 0.4985, 0.0006, 0.0005, 0.0004];
        let (tree, _) = chain_tree(&weights_, &[]);
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let w = weights_.map(|weight| -weight.ln());
        // The node's top falls in the first row, and the step between the
        // thick ones in the row its grid line starts.
        near_all(
            &spans(&frame, ROOT),
            &[
                (0.0, 1.0, 0.0, w[0]),
                (1.0, 400.0, w[0], w[0]),
                (400.0, 401.0, 0.0, w[1]),
                (401.0, 798.0, w[1], w[1]),
                (798.0, 799.0, 0.0, w[2]),
                (799.0, 800.0, 0.0, w[4]),
            ],
        );
        // Only the thick ones are boxes; a pixel of the thin selects the
        // child it holds.
        assert_eq!(frame.boxes.len(), 2);
        let hit = pick(&tree, LENS, &frame, &camera, VIEW, 100.0, 799.5, 1.0)
            .unwrap()
            .unwrap();
        assert_eq!((hit.r#box.branch, hit.exact), (Some(3), true));
    }

    #[test]
    fn children_of_no_known_width_are_a_hole() {
        // q has not answered: nothing is drawn, though p places the children.
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.5, 0.5])).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let frame = layout(&tree, lens, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        assert!(frame.spans.is_empty());
        assert_eq!(frame.boxes.len(), 2);
    }

    #[test]
    fn spans_are_whole_physical_rows_one_after_another() {
        // One heavy child, then thin known children alternating with
        // one-child gaps, about forty to a pixel.
        let w = 1.25e-5;
        let mut runs = vec![listed(0, &[0], &[0.99], 0.0)];
        for i in 0..400u32 {
            let below = 0.01 - 2.0 * f64::from(i) * w;
            runs.push(listed(1 + 2 * i, &[1 + 2 * i], &[w], 1.0 - below));
        }
        let mut tree = Tree::new();
        tree.insert(ROOT, 0, runs, None).unwrap();
        let camera = cam(ROOT, 0.0, top(&[]), 0.3 / VIEW.height);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let rows = spans(&frame, ROOT);
        assert!(2 < rows.len());
        for pair in rows.windows(2) {
            near(pair[0].1, pair[1].0);
        }
        for &(y0, y1, ..) in &rows {
            near(y0, y0.round());
            near(y1, y1.round());
        }
    }

    #[test]
    fn a_row_holding_a_thin_gap_uses_known_children_only() {
        // A heavy child to 798, a gap of four unknown children to 798.8, and
        // a child from there to the bottom: the gap is a hole and never
        // fabricates a wider width than the known children in the row.
        let mut tree = Tree::new();
        let runs = vec![
            listed(0, &[0], &[0.9975], 0.0),
            listed(5, &[5], &[0.0015], 0.9985),
        ];
        tree.insert(ROOT, 0, runs, Some(6)).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let (heavy, last) = (-0.9975f64.ln(), -0.0015f64.ln());
        let rows = spans(&frame, ROOT);
        assert_eq!(rows.len(), 4);
        near_all(
            &rows[..2],
            &[(0.0, 1.0, 0.0, heavy), (1.0, 798.0, heavy, heavy)],
        );
        near_all(&rows[2..3], &[(798.0, 799.0, 0.0, last)]);
        assert!((rows[3].2 - last).abs() < 1e-6 && (rows[3].3 - last).abs() < 1e-6);
    }

    #[test]
    fn a_thin_child_never_hides_a_thick_one_in_any_order() {
        // By branch: a heavy child, a child a tenth of a pixel tall, then
        // one of 80 pixels whose top is within a pixel of the thin one's.
        let mut tree = Tree::new();
        let runs = vec![
            listed(0, &[0], &[0.9], 0.0),
            listed(1, &[1, 2], &[1e-4, 0.0999], 0.9),
        ];
        tree.insert(ROOT, 0, runs, Some(3)).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let thick = frame
            .boxes
            .iter()
            .find(|b| b.branch == Some(2))
            .expect("the thick child is drawn as itself");
        near(thick.y1 - thick.y0, 0.0999);
    }

    #[test]
    fn a_row_pixel_selects_a_known_child_it_holds_over_a_gap() {
        // Known children alternate with one-child gaps, about 40 a pixel.
        let w = 1.25e-5;
        let mut runs = vec![listed(0, &[0], &[0.99], 0.0)];
        for i in 0..400u32 {
            let below = 0.01 - 2.0 * f64::from(i) * w;
            runs.push(listed(1 + 2 * i, &[1 + 2 * i], &[w], 1.0 - below));
        }
        let mut tree = Tree::new();
        tree.insert(ROOT, 0, runs, None).unwrap();
        // Off the grid, so pixel corners fall inside gaps, not on edges:
        // a child and a gap repeat every 0.02 of a pixel.
        let camera = cam(ROOT, 0.0, top(&[]), 0.385 / VIEW.height);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let selected = |y_px: f64, pixel_px: f64| {
            let hit = pick(&tree, LENS, &frame, &camera, VIEW, 100.0, y_px, pixel_px);
            hit.unwrap().unwrap().r#box.branch
        };
        let mut corners_in_gaps = 0;
        for row in 792..800 {
            let y_px = f64::from(row);
            assert!(selected(y_px, 1.0).is_some(), "row {row} selects a gap");
            corners_in_gaps += usize::from(selected(y_px, 1e-9).is_none());
        }
        assert!(0 < corners_in_gaps);
    }

    #[test]
    fn a_gap_drawn_as_its_own_box_keeps_its_pixels() {
        // The gap ends half a pixel into row 600, where the child after it
        // starts; that child's pixels start at row 601.
        let below = 1.0 - 600.5 / VIEW.height;
        let runs = vec![
            listed(0, &[0], &[0.5], 0.0),
            listed(5, &[5], &[0.1], 1.0 - below),
        ];
        let mut tree = Tree::new();
        tree.insert(ROOT, 0, runs, None).unwrap();
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let selected = |y_px: f64| {
            let hit = pick(&tree, LENS, &frame, &camera, VIEW, 100.0, y_px, 1.0);
            hit.unwrap().unwrap().r#box.kind
        };
        assert_eq!(selected(600.0), BoxKind::Gap);
        assert_eq!(selected(601.0), BoxKind::Leaf);
    }

    #[test]
    fn deep_anchors_stay_exact_beyond_absolute_float_precision() {
        // 80 halvings put the deepest node 2^-80 of the root tall.
        let path: Vec<Branch> = (0..80).map(|i| i % 2).collect();
        let (tree, deepest) = chain_tree(&[0.5, 0.5], &path);
        // Half way down the deepest node, in the middle of the view.
        let camera = cam(ROOT, 80.0 * LN_2, at(&path, &[0x80]), 0.5);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let mut children: Vec<_> = frame.boxes.iter().filter(|b| b.node == deepest).collect();
        children.sort_by_key(|b| b.rank);
        assert_eq!(children.len(), 2);
        near(children[0].y0, 0.0);
        near(children[0].y1, 0.5);
        near(children[1].y1, 1.0);
        // The deepest node fills the view: no better name for the point.
        assert_eq!(settle(&tree, LENS, &camera).unwrap(), camera.anchor);
    }

    #[test]
    fn a_pointer_deep_below_the_origin_picks_where_floats_of_the_origin_overflow() {
        // 1100 halvings: the origin is e^762 view heights tall, past any float.
        let path: Vec<Branch> = (0..1100).map(|i| i % 2).collect();
        let (tree, deepest) = chain_tree(&[0.5, 0.5], &path);
        let camera = cam(ROOT, 1100.0 * LN_2, at(&path, &[0x80]), 0.5);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let hit = pick(&tree, LENS, &frame, &camera, VIEW, 500.0, 600.0, 1.0)
            .unwrap()
            .unwrap();
        assert_eq!((hit.r#box.node, hit.r#box.branch), (deepest, Some(1)));
    }

    #[test]
    fn ancestors_past_any_float_of_the_view_are_laid_out_and_picked_through() {
        // p halves 1100 times; q gives each step 2^-16, so the boxes of
        // ancestors far taller than a float tells start right of the view.
        let path: Vec<Branch> = (0..1100).map(|i| i % 2).collect();
        let (mut tree, deepest) = chain_tree(&[0.5, 0.5], &path);
        let q = [2f64.powi(-16), 2f64.powi(-16), 1.0 - 2f64.powi(-15)];
        let mut node = ROOT;
        tree.set(node, 1, weights(&q)).unwrap();
        for &branch in &path {
            node = tree.child(node, branch).unwrap();
            tree.set(node, 1, weights(&q)).unwrap();
        }
        let lens = Lens { p: 0, q: 1 };
        let camera = cam(ROOT, 1100.0 * LN_2, at(&path, &[0x80]), 0.5);
        let frame = layout(&tree, lens, &camera, VIEW, 1.0).unwrap();
        // 5 nats right of the view's left edge, 767.4 nats right of the
        // root's, is the box of the 70th branch, e^714 view heights tall.
        let hit = pick(&tree, lens, &frame, &camera, VIEW, 500.0, 600.0, 1.0)
            .unwrap()
            .unwrap();
        let node = path[..69].iter().fold(ROOT, |node, &b| child_hash(node, b));
        assert_eq!((hit.r#box.node, hit.r#box.branch), (node, Some(path[69])));
        assert!(frame.boxes.iter().all(|b| b.node != deepest));
    }

    #[test]
    fn a_child_deep_in_a_tail_sits_exactly_beside_its_neighbours() {
        // Three children 2^-1000 of the root, below a heavy one: no float of
        // a cdf tells them apart, but a run's exact top does.
        let tiny = -1000.0 * LN_2;
        let above = Point::bottom().moved(false, 3.0, tiny).unwrap();
        let mut tree = Tree::new();
        let runs = vec![
            listed(0, &[0], &[1.0], 0.0),
            Run::weighed(1, &[1, 2, 3], &[tiny.exp(); 3], &above, 140),
        ];
        tree.insert(ROOT, 0, runs, Some(4)).unwrap();
        // The view is the tail itself: x = 1000 bits is where children
        // 2^-1000 of the root fill it, from the top of the first of them.
        let camera = cam(ROOT, -tiny - 3f64.ln(), at(&[], &above.digits()), 0.0);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let ys: Vec<_> = frame
            .boxes
            .iter()
            .filter(|b| b.rank > 0)
            .map(|b| (b.y0, b.y1))
            .collect();
        assert_eq!(ys.len(), 3);
        for (i, &(y0, y1)) in ys.iter().enumerate() {
            near(y0, i as f64 / 3.0);
            near(y1, (i + 1) as f64 / 3.0);
        }
    }

    #[test]
    fn gaps_have_unknown_width_and_rows_draw_from_their_samples() {
        let mut tree = Tree::new();
        // The two heaviest children of the root, and the node's end at rank
        // 1000: a quarter of its mass in 998 unknown children.
        let run = listed(0, &[4, 7], &[0.5, 0.25], 0.0);
        tree.insert(ROOT, 0, vec![run], Some(1000)).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let gap = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .expect("the unknown quarter is a gap");
        // Gaps never fabricate a width before their children arrive.
        assert_eq!((gap.rank, gap.count), (2, 998));
        assert!(gap.x1.is_infinite());
        assert!(gap.q.is_nan());
        near(gap.y0, 0.75);
        // Same when the count is unknown.
        let mut uncounted = Tree::new();
        let run = listed(0, &[4, 7], &[0.5, 0.25], 0.0);
        uncounted.insert(ROOT, 0, vec![run], None).unwrap();
        let frame_ = layout(&uncounted, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let gap_ = frame_
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .unwrap();
        assert!(gap_.x1.is_infinite());
        assert!(gap_.q.is_nan());
        // An answer of four samples splits the gap: each sample, thinner
        // than a pixel, is drawn in its row at its own width, while unsampled
        // rows between them stay empty.
        let samples: Vec<_> = (0..4)
            .map(|i| {
                let below = 0.25 - 0.0625 * f64::from(i);
                listed(2 + 100 * i, &[100 + i], &[1e-4], 1.0 - below)
            })
            .collect();
        tree.insert(ROOT, 0, samples, None).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        // Each sample's row reaches its own width.
        let width = -(1e-4f64).ln();
        let rows = spans(&frame, ROOT);
        for top in [600.0, 650.0, 700.0, 750.0] {
            let row = rows.iter().find(|row| row.0 <= top && top < row.1).unwrap();
            assert!(row.2 <= width && width <= row.3, "{row:?}");
        }
        assert!(rows.iter().all(|row| !(row.0 <= 625.0 && 625.0 < row.1)));
    }

    #[test]
    fn a_view_is_told_from_the_deepest_node_q_places() {
        // All of p's mass goes to branch 0, and q is unknown at the root: the
        // child holds the view, but where its children start is q's to say.
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[1.0])).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let camera = cam(ROOT, 0.0, at(&[0], &[]), 0.0);
        assert_eq!(view(&tree, lens, &camera, VIEW, 100.0).unwrap().node, ROOT);
        tree.set(ROOT, 1, weights(&[1.0])).unwrap();
        let child = tree.child(ROOT, 0).unwrap();
        assert_eq!(view(&tree, lens, &camera, VIEW, 100.0).unwrap().node, child);
    }

    #[test]
    fn a_view_no_float_of_a_node_resolves_is_told_exactly() {
        // A child 2^-80 of it in view, a quarter of the way down it: its
        // masses below the view's top and bottom are one float.
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.5, 0.5])).unwrap();
        let camera = cam(ROOT, 81.0 * LN_2, at(&[1], &[0x40]), 0.5);
        let seen = view(&tree, LENS, &camera, VIEW, 1600.0).unwrap();
        let child = tree.child(ROOT, 1).unwrap();
        // The view's top is half a view above the point, 0.25 − 2^-81, as
        // near as a mass of 2^-81 is told: 0x3f, nine 0xff, about 0x80.
        let mut top = vec![0x3f];
        top.extend([0xff; 9]);
        assert_eq!(
            (seen.node, &seen.top[..10], seen.rows),
            (child, &top[..], 1600.0)
        );
        assert!([0x7f, 0x80].contains(&seen.top[10]));
        near(seen.ln_height, -80.0 * LN_2);
        // The child's children start ln 2 right of the origin's.
        near(seen.right, camera.x + VIEW.width / camera.k - LN_2);
        // A source that knows the child places what the view shows: the
        // child's first branch, which holds the whole view.
        tree.set(child, 0, weights(&[0.5, 0.5])).unwrap();
        let mut session = crate::session::Session::new();
        let demand = crate::session::Demand {
            lens: LENS,
            view: Some(seen),
            path: None,
        };
        session.demand(&tree, demand).unwrap();
        let updates = session.step(&mut tree).unwrap();
        let [crate::session::Update::Answer { node, runs, .. }] = &updates[..] else {
            panic!("one answer: {updates:?}");
        };
        assert_eq!((*node, runs[0].branches().to_vec()), (child, vec![0]));
    }

    #[test]
    fn settling_moves_nothing_and_falls_back_to_what_p_places() {
        let path: Vec<Branch> = (0..30).map(|i| (i * 7) % 3).collect();
        let (tree, _) = chain_tree(&[0.6, 0.3, 0.1], &path);
        // A view a few nodes up the path from the point's own.
        let mut camera = cam(ROOT, 12.3, at(&path, &[0x51, 0x07]), 0.41);
        let before = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        camera.anchor = settle(&tree, LENS, &camera).unwrap();
        assert!(camera.anchor.path.len() < path.len());
        assert_eq!(camera.anchor.path, path[..camera.anchor.path.len()]);
        let after = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let key = |b: &LayoutBox| (b.node, b.rank, b.kind);
        let positions: HashMap<_, _> = before.boxes.iter().map(|b| (key(b), b.y0)).collect();
        let mut shared = 0;
        for b in &after.boxes {
            if let Some(&y0) = positions.get(&key(b)) {
                assert!((b.y0 - y0).abs() < 1e-9);
                shared += 1;
            }
        }
        assert!(shared > 0);
        // Settled, it stays as it is.
        assert_eq!(settle(&tree, LENS, &camera).unwrap(), camera.anchor);

        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.5, 0.5])).unwrap();
        tree.set(ROOT, 1, weights(&[1.0])).unwrap();
        let lost = cam(ROOT, 1.0, at(&[1], &[0x40]), 0.1);
        assert_eq!(settle(&tree, Lens { p: 1, q: 1 }, &lost).unwrap(), top(&[]));
        assert_eq!(settle(&tree, LENS, &lost).unwrap(), lost.anchor);
        assert!(layout(&tree, Lens { p: 1, q: 1 }, &lost, VIEW, 1.0).is_err());
    }

    #[test]
    fn zooming_out_and_back_in_returns_to_the_same_anchor() {
        let path: Vec<Branch> = (0..200).map(|i| (i * 7) % 3).collect();
        let (tree, _) = chain_tree(&[0.6, 0.3, 0.1], &path);
        // A point 150 nodes down, at a depth where its node about fills the
        // view, and views of it from there out to the root in 400 steps.
        let depth: f64 = path[..150]
            .iter()
            .map(|&branch| -[0.6f64, 0.3, 0.1][branch as usize].ln())
            .sum();
        let mut camera = cam(ROOT, depth, at(&path[..150], &[0x51, 0x07]), 0.37);
        let zoom = |camera: &mut Camera, out: bool| {
            let mut names = Vec::new();
            for step in 0..=400 {
                camera.x = depth * f64::from(if out { 400 - step } else { step }) / 400.0;
                camera.anchor = settle(&tree, LENS, camera).unwrap();
                // The point is where a zoom about it aims already.
                assert_eq!(
                    aim(&tree, LENS, camera, 0.37).unwrap().anchor,
                    camera.anchor
                );
                names.push(camera.anchor.clone());
            }
            names
        };
        camera.anchor = settle(&tree, LENS, &camera).unwrap();
        let deep = camera.anchor.clone();
        assert!(100 < deep.path.len());
        // Out to the root, every digit kept, and in again: every name on the
        // way matches the one the same view had on the way out.
        let mut out = zoom(&mut camera, true);
        assert_eq!(camera.anchor.path, [] as [Branch; 0]);
        assert!(20 < camera.anchor.digits.len());
        out.reverse();
        assert_eq!(zoom(&mut camera, false), out);
        assert_eq!(camera.anchor, deep);

        // Aiming elsewhere keeps the picture and drops what the view cannot show.
        camera.x = 20.0;
        camera.anchor = settle(&tree, LENS, &camera).unwrap();
        let before = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let aimed = aim(&tree, LENS, &camera, 0.6).unwrap();
        (camera.anchor, camera.a) = (aimed.anchor, aimed.a);
        assert!(camera.anchor.digits.len() < 16);
        let after = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        assert_eq!(before.boxes.len(), after.boxes.len());
        for (b, a) in before.boxes.iter().zip(&after.boxes) {
            assert_eq!((b.node, b.rank), (a.node, a.rank));
            assert!((b.y0 - a.y0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_view_across_an_edge_is_exact_on_both_sides_at_any_depth() {
        // The root halves; above the edge between its children the path
        // follows each node's last child, below it each node's first.
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.5, 0.5])).unwrap();
        let mut ends = Vec::new();
        for side in [1, 0] {
            let mut node = tree.child(ROOT, 1 - side).unwrap();
            for _ in 0..80 {
                tree.set(node, 0, weights(&[0.5, 0.5])).unwrap();
                node = tree.child(node, side).unwrap();
            }
            tree.set(node, 0, weights(&[0.5, 0.5])).unwrap();
            ends.push((node, side));
        }
        // The edge across the middle of a view √2 the height of the
        // children 2^-82 of the root that meet at it.
        let mut camera = cam(ROOT, 81.5 * LN_2, at(&[], &[0x80]), 0.5);
        camera.anchor = settle(&tree, LENS, &camera).unwrap();
        // No child of the root covers the view: the point stays the root's.
        assert_eq!(camera.anchor, at(&[], &[0x80]));
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let height = 0.5f64.sqrt();
        for ((node, side), y0) in ends.into_iter().zip([0.5 - height, 0.5]) {
            let b = frame
                .boxes
                .iter()
                .find(|b| b.node == node && b.branch == Some(side))
                .expect("the child beside the edge is in view");
            near(b.y0, y0);
            near(b.y1, y0 + height);
        }
        // A point 2^-90 of the root off the edge is a point of one side,
        // named by the deepest node there that covers the view.
        let mut digits = vec![0x80; 1];
        digits.extend([0; 10]);
        digits.push(0x40);
        camera.anchor = at(&[], &digits);
        camera.x = 100.0 * LN_2;
        camera.anchor = settle(&tree, LENS, &camera).unwrap();
        assert_eq!(camera.anchor.path.len(), 82);
        assert_eq!(camera.anchor.path[0], 1);
    }

    #[test]
    fn the_view_clamps_inside_the_root_and_picks_beyond_it_find_the_origin() {
        let (tree, _) = chain_tree(&[0.5, 0.5], &[0]);
        let camera = cam(ROOT, 0.5, top(&[0]), 0.4);
        let a = clamp_position(&tree, LENS, &camera).unwrap();
        near(a, 0.0);
        near(
            extent(&tree, LENS, &Camera { a, ..camera }, 0).unwrap().top,
            0.0,
        );

        let empty = Tree::new();
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&empty, LENS, &camera, VIEW, 1.0).unwrap();
        let hit = pick(&empty, LENS, &frame, &camera, VIEW, 500.0, 400.0, 1.0)
            .unwrap()
            .unwrap();
        assert_eq!((hit.r#box.node, hit.exact), (ROOT, false));
        assert_eq!(
            pick(&empty, LENS, &frame, &camera, VIEW, -1.0, 400.0, 1.0).unwrap(),
            None
        );
    }

    #[test]
    fn past_a_node_with_no_children_the_pointer_picks_that_node() {
        // Branch 0 ends the sequence: nothing lies past its box.
        let (mut tree, _) = chain_tree(&[0.5, 0.5], &[]);
        let ended = tree.child(ROOT, 0).unwrap();
        tree.set(ended, 0, Distribution::from_ordered(Vec::new()).unwrap())
            .unwrap();
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let hit = pick(&tree, LENS, &frame, &camera, VIEW, 300.0, 200.0, 1.0)
            .unwrap()
            .unwrap();
        assert_eq!(
            (hit.r#box.node, hit.r#box.branch, hit.exact),
            (ROOT, Some(0), true)
        );
    }
}
