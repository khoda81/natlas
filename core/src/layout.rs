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
use std::ops::Range;

use serde::{Deserialize, Serialize};

use std::f64::consts::LN_2;

use crate::distribution::{Distribution, Edge, Item, Rank};
use crate::point::Point;
use crate::session::View;
use crate::tree::{bytes, child_hash, node_id, Branch, Lens, NodeId, Tree};
use crate::Result;

/// View heights within which a node's edge is at the view's.
const FIT: f64 = 1e-12;
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

    /// The point at a view height, clamped to the node.
    fn at(&self, y: f64) -> Point {
        let (edge, at) = if y <= self.top.1 || y - self.top.1 <= self.bottom.1 - y {
            &self.top
        } else {
            &self.bottom
        };
        edge.moved(*at <= y, (y - at).abs(), -self.ln_scale)
            .unwrap_or_else(|| {
                if *at <= y {
                    Point::bottom()
                } else {
                    Point::top()
                }
            })
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
struct Heights<'a> {
    placed: &'a Placed,
    d: &'a Distribution,
    /// View heights per unit of the node's mass, where a float holds it.
    scale: f64,
    /// Each run's point's height, NaN until asked.
    runs: Vec<f64>,
    /// The last edge asked, and its height: the next item's top is the
    /// bottom of the one before.
    last: Option<(Edge, f64)>,
}

impl<'a> Heights<'a> {
    fn new(placed: &'a Placed, d: &'a Distribution) -> Self {
        Self {
            placed,
            d,
            scale: placed.ln_scale.exp(),
            runs: vec![f64::NAN; d.runs().len()],
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
            if let Some((point, offsets)) = self.d.runs()[run].offsets() {
                if self.runs[run].is_nan() {
                    self.runs[run] = self.placed.y(point);
                }
                if self.runs[run].abs() <= NEAR {
                    let offset = if self.scale.is_finite() {
                        offsets[at] * self.scale
                    } else {
                        (offsets[at].ln() + self.placed.ln_scale).exp()
                    };
                    return self.runs[run] + offset;
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
    pub placed: HashMap<NodeId, Placed>,
    pub origin_top: f64,
    pub origin_bottom: f64,
}

impl Frame {
    /// Empty, keeping the allocations for the next view.
    pub(crate) fn clear(&mut self) {
        self.boxes.clear();
        self.spans.clear();
        self.placed.clear();
        self.origin_top = 0.0;
        self.origin_bottom = 0.0;
    }
}

/// A child's or a gap's box, as laid out and as picked.
fn item_box(
    tree: &Tree,
    lens: Lens,
    node: NodeId,
    placed: &Placed,
    d: &Distribution,
    item: Item,
    (y0, y1): (f64, f64),
) -> Result<LayoutBox> {
    let rank = d.item_rank(item);
    // A child is sized by its weight, a gap by what its neighbours leave it.
    let ln_p = match item {
        Item::Child { run, at } => d.runs()[run].ln_p(at),
        Item::Gap { .. } => {
            let (top, bottom) = d.item_edges(item);
            Point::ln_between(&top, &bottom)
        }
    };
    let (kind, branch, nats) = match item {
        Item::Child { run, at } => {
            let branch = d.runs()[run].branch(at);
            let expanded = tree
                .distribution(child_hash(node, branch), lens.p)?
                .is_some();
            let kind = if expanded {
                BoxKind::Node
            } else {
                BoxKind::Leaf
            };
            (kind, Some(branch), tree.nats_q(node, lens, branch, ln_p)?)
        }
        Item::Gap { before } => {
            let nats = gap_nats(tree, lens, node, d, before, ln_p, rank, d.item_end(item))?;
            (BoxKind::Gap, None, nats)
        }
    };
    Ok(LayoutBox {
        node,
        rank,
        count: d.item_end(item).map_or(0, |end| end - rank),
        kind,
        branch,
        p: ln_p.exp(),
        q: if nats.is_finite() {
            (-nats).exp()
        } else {
            f64::NAN
        },
        ln_mass: placed.ln_mass + ln_p,
        x0: placed.x,
        x1: placed.x + nats,
        y0,
        y1,
    })
}

/// The surprisal of a gap's children, as well as what is known tells, until
/// samples tell more: with q = p and their number known, that of their mean
/// weight, since children heaviest first are nearly alike side by side;
/// otherwise the mean surprisal of the known children beside the gap.
/// Infinite with neither.
#[allow(clippy::too_many_arguments)]
fn gap_nats(
    tree: &Tree,
    lens: Lens,
    node: NodeId,
    d: &Distribution,
    before: usize,
    ln_mass: f64,
    rank: Rank,
    end: Option<Rank>,
) -> Result<f64> {
    if let (true, Some(end)) = (lens.p == lens.q, end) {
        if rank < end {
            return Ok(f64::from(end - rank).ln() - ln_mass);
        }
    }
    let above = before
        .checked_sub(1)
        .map(|run| &d.runs()[run])
        .filter(|run| !run.is_empty())
        .map(|run| (run, run.len() - 1));
    let below = d
        .runs()
        .get(before)
        .filter(|run| !run.is_empty())
        .map(|run| (run, 0));
    let (mut sum, mut known) = (0.0, 0.0);
    for (run, at) in above.into_iter().chain(below) {
        let nats = tree.nats_q(node, lens, run.branch(at), run.ln_p(at))?;
        if nats.is_finite() {
            (sum, known) = (sum + nats, known + 1.0);
        }
    }
    Ok(if 0.0 < known {
        sum / known
    } else {
        f64::INFINITY
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
    xs: Vec<f64>,
}

fn steps(tree: &Tree, lens: Lens, origin: NodeId, path: &[Branch]) -> Result<Steps> {
    let mut steps = Steps {
        nodes: vec![origin],
        entries: Vec::with_capacity(path.len()),
        edges: Vec::with_capacity(path.len()),
        ln_masses: vec![0.0],
        xs: vec![0.0],
    };
    for (i, &branch) in path.iter().enumerate() {
        let node = steps.nodes[i];
        let Some(p) = tree.distribution(node, lens.p)? else {
            break;
        };
        let Some((rank, (run, at))) = p
            .rank(branch)
            .and_then(|rank| Some((rank, p.locate_rank(rank)?)))
        else {
            break;
        };
        let (top, bottom) = p.item_edges(Item::Child { run, at });
        // Placed by its extent, sized by its weight.
        let ln_extent = Point::ln_between(&top, &bottom);
        let ln_p = p.runs()[run].ln_p(at);
        let nats_q = tree.nats_q(node, lens, branch, ln_p)?;
        steps.nodes.push(child_hash(node, branch));
        steps.entries.push(ChainEntry {
            branch,
            rank,
            ln_p,
            nats_q,
        });
        steps.edges.push((top, bottom));
        steps.ln_masses.push(steps.ln_masses[i] + ln_extent);
        steps.xs.push(steps.xs[i] + nats_q);
    }
    Ok(steps)
}

/// The known edges nearest a point of a node, above and below it: those of
/// the item holding it, or the node's own.
fn around(d: Option<&Distribution>, point: &Point) -> (Point, Point) {
    d.and_then(|d| Some(d.item_edges(d.locate(point)?)))
        .filter(|(top, bottom)| point.within(top, bottom))
        .unwrap_or_else(|| (Point::top(), Point::bottom()))
}

/// The known child of a node holding a point, and its edges.
fn held(d: Option<&Distribution>, point: &Point) -> Option<(Branch, Point, Point)> {
    let d = d?;
    let item = d.locate(point)?;
    let Item::Child { run, at } = item else {
        return None;
    };
    let (top, bottom) = d.item_edges(item);
    point
        .within(&top, &bottom)
        .then(|| (d.runs()[run].branch(at), top, bottom))
}

/// Bits that tell a point of a node e^`ln_scale` view heights tall to
/// 2^-64 of a view height.
fn precision(ln_scale: f64) -> u64 {
    (64.0 + ln_scale / LN_2).ceil().max(0.0) as u64
}

/// The nodes from the origin down the anchor's path, each one's entry in its
/// parent, where each is placed and its own extent in view heights: the
/// anchor's node around its point, every ancestor at its chain child's edges.
struct Chain {
    nodes: Vec<NodeId>,
    entries: Vec<ChainEntry>,
    placed: Vec<Placed>,
    spans: Vec<(f64, f64)>,
}

fn place_chain(tree: &Tree, lens: Lens, cam: &Camera) -> Result<Chain> {
    let Steps {
        nodes,
        entries,
        edges,
        ln_masses,
        xs,
    } = steps(tree, lens, cam.origin, &cam.anchor.path)?;
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
    let (top, bottom) = around(tree.distribution(nodes[depth], lens.p)?, &point);
    let (y_top, y_bottom) = (y(&top), y(&bottom));
    let anchor = Placed::new(
        (top, y_top),
        (bottom, y_bottom),
        ln_masses[depth],
        ln_scale,
        xs[depth],
    );
    let mut placed = vec![anchor; nodes.len()];
    let mut spans = vec![(y(&Point::top()), y(&Point::bottom())); nodes.len()];
    for (i, (top, bottom)) in edges.into_iter().enumerate().rev() {
        let (y0, y1) = spans[i + 1];
        placed[i] = Placed::new(
            (top, y0),
            (bottom, y1),
            ln_masses[i],
            ln_masses[i] + cam.x,
            xs[i],
        );
        spans[i] = placed[i].span();
    }
    Ok(Chain {
        nodes,
        entries,
        placed,
        spans,
    })
}

struct Layout<'a> {
    tree: &'a Tree,
    lens: Lens,
    cam: &'a Camera,
    right: f64,
    /// A physical row, in view heights: a child shorter than one is drawn
    /// in the rows it falls in.
    row: f64,
    frame: Frame,
}

impl Layout<'_> {
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
        let mut heights = Heights::new(&placed, d);
        let mut raster = Raster::new(self.row);
        // From the first item that ends in the row above the view, so the
        // step above the first in it is known, to the last that starts in it.
        let mut item = d.locate(&placed.at(-self.row));
        while let Some(found) = item {
            item = d.next(found);
            let (y0, y1) = heights.item(found);
            if 1.0 <= y0 {
                break;
            }
            // How wide it is drawn; a gap of children no width is known for
            // is a hole in the node's children.
            let width = match found {
                Item::Child { run, at } => {
                    let r = &d.runs()[run];
                    tree.nats_q(node, self.lens, r.branch(at), r.ln_p(at))?
                }
                Item::Gap { before } => {
                    let ln_mass = (y1 - y0).ln() - placed.ln_scale;
                    let (rank, end) = (d.item_rank(found), d.item_end(found));
                    gap_nats(tree, self.lens, node, d, before, ln_mass, rank, end)?
                }
            };
            raster.add(y0, y1, width);
            let chain = matches!(found, Item::Child { .. }) && skip == Some(d.item_rank(found));
            if chain || y1 - y0 < self.row {
                continue;
            }
            let b = item_box(tree, self.lens, node, &placed, d, found, (y0, y1))?;
            match found {
                Item::Child { .. } => self.child(node, &placed, d, found, b)?,
                Item::Gap { .. } => self.push(b),
            }
        }
        for span in raster.finish() {
            self.span(node, &placed, span);
        }
        Ok(())
    }

    /// A span of a node's rows, where it shows.
    fn span(&mut self, node: NodeId, placed: &Placed, span: RowSpan) {
        let (y0, y1) = (
            span.rows.start as f64 * self.row,
            span.rows.end as f64 * self.row,
        );
        let reach = placed.x + span.reach;
        if y1 <= 0.0 || 1.0 <= y0 || reach <= self.cam.x || self.right <= placed.x {
            return;
        }
        self.frame.spans.push(Span {
            node,
            y0,
            y1,
            x0: placed.x,
            fill: placed.x + span.fill.unwrap_or(0.0),
            reach,
        });
    }

    /// A child's box, and its children where they start in view.
    fn child(
        &mut self,
        node: NodeId,
        placed: &Placed,
        d: &Distribution,
        item: Item,
        b: LayoutBox,
    ) -> Result<()> {
        let id = child_hash(node, b.branch.expect("a child has a branch"));
        let (x1, kind, ys) = (b.x1, b.kind, (b.y0, b.y1));
        self.push(b);
        if self.right <= x1 || kind != BoxKind::Node {
            return Ok(());
        }
        let (top, bottom) = d.item_edges(item);
        let ln_extent = Point::ln_between(&top, &bottom);
        self.lay(id, placed.child(ys, ln_extent, x1), None)
    }
}

/// A run of physical rows of a node's children drawn alike, in the rows' own
/// units: down from `rows.start`, filled as far as `fill`, before which no
/// edge of them falls in any of the rows, and outlined to `reach`, where the
/// last edge in them falls. Without `fill`, an edge reaching back to where
/// the children start falls in the rows: they are outline throughout.
#[derive(Clone, Debug, PartialEq)]
struct RowSpan {
    rows: Range<i64>,
    fill: Option<f64>,
    reach: f64,
}

/// A node's children as physical rows, each pixel the half-open square
/// from its corner: one an edge of them falls in is outline, one wholly
/// inside them is filled. An edge on a grid line falls in the pixel below
/// it or to its right. In a row, the edges are each child's end, and each
/// step from one child to the next, which runs from the one's width to the
/// other's, or from where the children start at the node's top, its
/// bottom, and the sides of a hole: every pixel from the first edge to the
/// last holds one, since the steps chain from one width to the next.
struct Raster {
    /// A physical row, in view heights.
    height: f64,
    /// The width of the item before, none for a hole, and where it ends,
    /// in rows; none before the first.
    before: Option<(Option<f64>, f64)>,
    open: Option<RowEdges>,
    spans: Vec<RowSpan>,
}

/// The nearest and furthest edges in one row.
struct RowEdges {
    index: i64,
    near: f64,
    far: f64,
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
        // The step at its top, from the item before, or from where the
        // children start at the node's top or below a hole.
        let before = self.before.and_then(|(width, _)| width).unwrap_or(0.0);
        self.edge(first, before);
        self.edge(first, width.unwrap_or(0.0));
        if let Some(width) = width {
            // The rows between, wholly its own: no edge but its end.
            if first + 1 < last {
                self.flush();
                self.emit(RowSpan {
                    rows: first + 1..last,
                    fill: Some(width),
                    reach: width,
                });
            }
            self.edge(last, width);
        }
        self.before = Some((width, bottom));
    }

    /// An edge in row `index` reaching `width` from where the children start.
    fn edge(&mut self, index: i64, width: f64) {
        if self.open.as_ref().is_some_and(|row| row.index != index) {
            self.flush();
        }
        let row = self.open.get_or_insert(RowEdges {
            index,
            near: width,
            far: width,
        });
        row.near = row.near.min(width);
        row.far = row.far.max(width);
    }

    fn flush(&mut self) {
        if let Some(row) = self.open.take() {
            self.emit(RowSpan {
                rows: row.index..row.index + 1,
                fill: (0.0 < row.near).then_some(row.near),
                reach: row.far,
            });
        }
    }

    /// A span, joined to the one before where it continues it alike.
    fn emit(&mut self, span: RowSpan) {
        if let Some(last) = self.spans.last_mut() {
            if last.rows.end == span.rows.start
                && last.fill == span.fill
                && last.reach == span.reach
            {
                last.rows.end = span.rows.end;
                return;
            }
        }
        self.spans.push(span);
    }

    /// The step at the bottom of the last item, to where the children start.
    fn finish(mut self) -> Vec<RowSpan> {
        if let Some((width, bottom)) = self.before {
            let row = Self::row(bottom);
            self.edge(row, width.unwrap_or(0.0));
            self.edge(row, 0.0);
        }
        self.flush();
        self.spans
    }
}

pub fn layout(tree: &Tree, lens: Lens, cam: &Camera, view: Viewport, row_px: f64) -> Result<Frame> {
    layout_into(tree, lens, cam, view, row_px, Frame::default())
}

pub(crate) fn layout_into(
    tree: &Tree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    row_px: f64,
    mut frame: Frame,
) -> Result<Frame> {
    frame.clear();
    let Chain {
        nodes,
        entries,
        placed,
        spans,
    } = place_chain(tree, lens, cam)?;
    for (&node, placed) in nodes.iter().zip(&placed) {
        frame.placed.insert(node, placed.clone());
    }
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
    tree: &Tree,
    lens: Lens,
    frame: &Frame,
    node: NodeId,
) -> Result<Vec<(Branch, f64, f64)>> {
    let (Some(placed), Some(d)) = (frame.placed.get(&node), tree.distribution(node, lens.p)?)
    else {
        return Ok(Vec::new());
    };
    let mut heights = Heights::new(placed, d);
    let mut known = Vec::new();
    let mut item = d.locate(&placed.at(0.0));
    while let Some(found) = item {
        let (y0, y1) = heights.item(found);
        if 1.0 <= y0 {
            break;
        }
        if let Item::Child { run, at } = found {
            known.push((d.runs()[run].branch(at), y0, y1));
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
pub fn view(tree: &Tree, lens: Lens, cam: &Camera, viewport: Viewport, rows: f64) -> Result<View> {
    let Chain {
        nodes,
        placed,
        spans,
        ..
    } = place_chain(tree, lens, cam)?;
    let right = cam.x + viewport.width / cam.k;
    let covers = |&(top, bottom): &(f64, f64)| top <= 0.0 && 1.0 <= bottom;
    let (depth, top, ln_height, rows) = match spans.iter().rposition(covers) {
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
    Ok(View {
        node: nodes[depth],
        top: top.digits(),
        ln_height,
        rows,
        right: right - placed[depth].x,
    })
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

pub fn reach(tree: &Tree, lens: Lens, origin: NodeId, path: &[Branch]) -> Result<Reach> {
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
pub fn extent(tree: &Tree, lens: Lens, cam: &Camera, depth: usize) -> Result<Extent> {
    let spans = place_chain(tree, lens, cam)?.spans;
    let &(top, bottom) = spans.get(depth).ok_or_else(|| {
        format!(
            "the camera's path is placed {} deep, not {depth}",
            spans.len() - 1
        )
    })?;
    Ok(Extent { top, bottom })
}

/// Keep the view inside the root's interval, returning only the new anchor offset.
pub fn clamp_position(tree: &Tree, lens: Lens, cam: &Camera) -> Result<f64> {
    // Only the root's interval is all there is; beyond any other origin lies
    // its parent, unknown until fetched.
    if cam.origin != crate::tree::ROOT {
        return Ok(cam.a);
    }
    let Extent { top, bottom } = extent(tree, lens, cam, 0)?;
    Ok(if top > 0.0 {
        cam.a - top
    } else if bottom < 1.0 {
        cam.a + 1.0 - bottom
    } else {
        cam.a
    })
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
pub fn settle(tree: &Tree, lens: Lens, cam: &Camera) -> Result<Anchor> {
    let steps = steps(tree, lens, cam.origin, &cam.anchor.path)?;
    let mut path = cam.anchor.path[..steps.entries.len()].to_vec();
    let mut point = if path.len() == cam.anchor.path.len() {
        Point::new(&cam.anchor.digits)
    } else {
        Point::new(&[])
    };
    // Whether the part of a node between two of its edges covers the view,
    // the node e^ln_scale view heights tall. A node that fits the view
    // covers it, whichever way its edges round.
    let covers = |point: &Point, ln_scale: f64, top: &Point, bottom: &Point| {
        let ((_, above), (_, under)) = (point.to(top), point.to(bottom));
        cam.a - (ln_scale + above).exp() <= FIT && 1.0 - FIT <= cam.a + (ln_scale + under).exp()
    };
    while let Some((top, bottom)) = path.len().checked_sub(1).map(|i| &steps.edges[i]) {
        let ln_scale = steps.ln_masses[path.len()] + cam.x;
        if covers(&point, ln_scale, &Point::top(), &Point::bottom()) {
            break;
        }
        point = point.out_of_child(top, bottom);
        path.pop();
    }
    let mut node = steps.nodes[path.len()];
    let mut ln_scale = steps.ln_masses[path.len()] + cam.x;
    while let Some((branch, top, bottom)) = held(tree.distribution(node, lens.p)?, &point) {
        if !covers(&point, ln_scale, &top, &bottom) {
            break;
        }
        ln_scale += Point::ln_between(&top, &bottom);
        point = point.into_child(&top, &bottom, precision(ln_scale));
        path.push(branch);
        node = child_hash(node, branch);
    }
    Ok(Anchor {
        path,
        digits: point.digits(),
    })
}

/// The camera's point moved to what the view shows `fy` view heights down,
/// to 2^-64 of a view height, under its settled name: the point a zoom about
/// `fy` then leaves where it is, so only depth changes. Unchanged where the
/// point is there already, or where the origin does not reach that far.
pub fn aim(tree: &Tree, lens: Lens, cam: &Camera, fy: f64) -> Result<Aim> {
    let same = Aim {
        anchor: cam.anchor.clone(),
        a: cam.a,
    };
    let by = fy - cam.a;
    if by == 0.0 || !by.is_finite() {
        return Ok(same);
    }
    let Anchor { path, digits } = settle(tree, lens, cam)?;
    let steps = steps(tree, lens, cam.origin, &path)?;
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
    tree: &Tree,
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
    };
    if !(frame.origin_top..frame.origin_bottom).contains(&y) || x < 0.0 {
        return Ok(Some(origin));
    }
    let mut node = cam.origin;
    // The box of the node the pointer is in, below the origin.
    let mut within = None;
    loop {
        let (Some(placed), Some(d)) = (frame.placed.get(&node), tree.distribution(node, lens.p)?)
        else {
            return Ok(Some(origin));
        };
        // Nothing at any height: a node known to have no children, whose
        // box is all there is.
        let Some(item) = d.locate(&placed.at(y)) else {
            return Ok(Some(within.map_or(origin, |b| PickResult {
                r#box: b,
                exact: true,
            })));
        };
        let mut heights = Heights::new(placed, d);
        let item = pixel_child(&mut heights, item, y, y + pixel_px / view.height);
        let b = item_box(tree, lens, node, placed, d, item, heights.item(item))?;
        let Some(branch) = b.branch.filter(|_| x >= b.x1) else {
            let exact = b.branch.is_some();
            return Ok(Some(PickResult { r#box: b, exact }));
        };
        let child = child_hash(node, branch);
        if !frame.placed.contains_key(&child) || tree.distribution(child, lens.p)?.is_none() {
            return Ok(Some(PickResult {
                r#box: b,
                exact: false,
            }));
        }
        within = Some(b);
        node = child;
    }
}

/// The item a pixel from `y` down to `bottom` selects, its corner in
/// `item`: a gap thinner than the pixel is drawn in the row it falls in,
/// with the children there, so the known child after it in the pixel is
/// selected instead. A gap a pixel or more tall is drawn as its own box and
/// keeps its pixels.
fn pixel_child(heights: &mut Heights, item: Item, y: f64, bottom: f64) -> Item {
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
    use crate::distribution::{Distribution, Order, Run, Scale};
    use crate::tree::ROOT;
    use std::f64::consts::LN_2;

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
        // thinner than a pixel: a row is filled as far as its narrowest
        // child reaches, and outlined to its widest.
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
                (400.0, 401.0, w[0], w[1]),
                (401.0, 798.0, w[1], w[1]),
                (798.0, 799.0, w[1], w[2]),
                (799.0, 800.0, w[2], w[4]),
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
        tree.insert(ROOT, 0, runs).unwrap();
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
    fn a_row_holding_a_thin_gap_takes_its_estimated_width() {
        // A heavy child to 798, a gap of four unknown children to 798.8, and
        // a child from there to the bottom.
        let mut tree = Tree::new();
        let runs = vec![
            listed(0, &[0], &[0.9975], 0.0),
            listed(5, &[5], &[0.0015], 0.9985),
            Run::end_at(6),
        ];
        tree.insert(ROOT, 0, runs).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        // The gap's children at their mean weight, as a gap box draws them.
        let (heavy, gap, last) = (-0.9975f64.ln(), (4.0f64 / 0.001).ln(), -0.0015f64.ln());
        let rows = spans(&frame, ROOT);
        assert_eq!(rows.len(), 4);
        near_all(
            &rows[..2],
            &[(0.0, 1.0, 0.0, heavy), (1.0, 798.0, heavy, heavy)],
        );
        // The row from 798 holds the step from the heavy child to the gap,
        // and from the gap to the last child.
        near_all(&rows[2..3], &[(798.0, 799.0, heavy, gap)]);
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
            Run::end_at(3),
        ];
        tree.insert(ROOT, 0, runs).unwrap();
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
        tree.insert(ROOT, 0, runs).unwrap();
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
        tree.insert(ROOT, 0, runs).unwrap();
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
    fn a_child_deep_in_a_tail_sits_exactly_beside_its_neighbours() {
        // Three children 2^-1000 of the root, below a heavy one: no float of
        // a cdf tells them apart, but a run's exact top does.
        let tiny = -1000.0 * LN_2;
        let above = Point::bottom().moved(false, 3.0, tiny).unwrap();
        let mut tree = Tree::new();
        let runs = vec![
            listed(0, &[0], &[1.0], 0.0),
            Run::weighed(1, &[1, 2, 3], &[tiny.exp(); 3], &above, 140),
            Run::end_at(4),
        ];
        tree.insert(ROOT, 0, runs).unwrap();
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
    fn gaps_draw_at_their_mean_weight_and_rows_from_their_samples() {
        let mut tree = Tree::new();
        // The two heaviest children of the root, and the node's end at rank
        // 1000: a quarter of its mass in 998 unknown children.
        let run = listed(0, &[4, 7], &[0.5, 0.25], 0.0);
        tree.insert(ROOT, 0, vec![run, Run::end_at(1000)]).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let gap = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .expect("the unknown quarter is a gap");
        // Until sampled, its 998 children are drawn at their mean weight.
        assert_eq!((gap.rank, gap.count), (2, 998));
        near(gap.x1 - gap.x0, (998.0f64 / 0.25).ln());
        near(gap.y0, 0.75);
        // Not knowing how many, it is drawn as the child above it.
        let mut uncounted = Tree::new();
        let run = listed(0, &[4, 7], &[0.5, 0.25], 0.0);
        uncounted.insert(ROOT, 0, vec![run]).unwrap();
        let frame_ = layout(&uncounted, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let gap_ = frame_
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .unwrap();
        near(gap_.x1 - gap_.x0, 4f64.ln());
        // An answer of four samples splits the gap: each sample, thinner
        // than a pixel, is drawn in its row at its own width.
        let samples: Vec<_> = (0..4)
            .map(|i| {
                let below = 0.25 - 0.0625 * f64::from(i);
                listed(2 + 100 * i, &[100 + i], &[1e-4], 1.0 - below)
            })
            .collect();
        tree.insert(ROOT, 0, samples).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        // Each sample's row reaches its own width.
        let width = -(1e-4f64).ln();
        let rows = spans(&frame, ROOT);
        for top in [600.0, 650.0, 700.0, 750.0] {
            let row = rows.iter().find(|row| row.0 <= top && top < row.1).unwrap();
            assert!(row.2 <= width && width <= row.3, "{row:?}");
        }
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
        tree.set(
            ended,
            0,
            Distribution::from_weights(&[], Order::Heaviest, Scale::Relative).unwrap(),
        )
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
