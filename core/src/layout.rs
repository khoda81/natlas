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

use serde::{Deserialize, Serialize};

use std::f64::consts::LN_2;

use crate::distribution::{ln_add, ln_between, Distribution, Item, Rank};
use crate::point::Point;
use crate::session::View;
use crate::tree::{bytes, child_hash, node_id, Branch, Lens, NodeId, Tree};
use crate::Result;

/// Known children of a run a band reads q from, when q is another measure.
const SAMPLES: usize = 4;
/// View heights within which a node's edge is at the view's.
const FIT: f64 = 1e-12;

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
    /// Consecutive children too thin to draw apart.
    Merged,
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
    /// The child's branch; none for a gap or a merged band.
    pub branch: Option<Branch>,
    /// Conditional probabilities; p-weighted geometric means for a band.
    pub p: f64,
    /// NaN while unknown; like p, a representative child weight for a band.
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

/// Where a laid-out node sits: two of its edges, as ln of the mass below
/// them, placed exactly in view heights.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    top: (f64, f64),
    bottom: (f64, f64),
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
    fn new(top: (f64, f64), bottom: (f64, f64), ln_mass: f64, ln_scale: f64, x: f64) -> Self {
        Self {
            top,
            bottom,
            ln_heights: (top.1.abs().ln(), bottom.1.abs().ln()),
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
    fn y(&self, ln_below: f64) -> f64 {
        if self.top.0 <= ln_below {
            return self.top.1 - (self.ln_scale + ln_between(ln_below, self.top.0)).exp();
        }
        if ln_below <= self.bottom.0 {
            return self.bottom.1 + (self.ln_scale + ln_between(self.bottom.0, ln_below)).exp();
        }
        let down = self.ln_scale + ln_between(self.top.0, ln_below);
        let up = self.ln_scale + ln_between(ln_below, self.bottom.0);
        if down.max(self.ln_heights.0) <= up.max(self.ln_heights.1) {
            self.top.1 + down.exp()
        } else {
            self.bottom.1 - up.exp()
        }
    }

    /// The edge at a view height, clamped to the node.
    fn ln_below(&self, y: f64) -> f64 {
        let (edge, at) = if y <= self.top.1 || y - self.top.1 <= self.bottom.1 - y {
            self.top
        } else {
            self.bottom
        };
        if y == at {
            return edge;
        }
        let ln_mass = (y - at).abs().ln() - self.ln_scale;
        if y < at {
            ln_add(edge, ln_mass).min(0.0)
        } else if ln_mass >= edge {
            f64::NEG_INFINITY
        } else {
            ln_between(edge, ln_mass)
        }
    }

    /// The node's own extent in view heights.
    fn span(&self) -> (f64, f64) {
        (self.y(0.0), self.y(f64::NEG_INFINITY))
    }

    /// Placed within a child of `p` mass between edges `top` and `bottom`.
    fn child(&self, top: f64, bottom: f64, ln_p: f64, x: f64) -> Placed {
        Placed::new(
            (0.0, self.y(top)),
            (f64::NEG_INFINITY, self.y(bottom)),
            self.ln_mass + ln_p,
            self.ln_scale + ln_p,
            x,
        )
    }
}

#[derive(Debug, Default)]
pub struct Frame {
    pub boxes: Vec<LayoutBox>,
    /// Every node laid out, for picking.
    pub placed: HashMap<NodeId, Placed>,
    pub origin_top: f64,
    pub origin_bottom: f64,
}

impl Frame {
    /// Empty, keeping the allocations for the next view.
    pub(crate) fn clear(&mut self) {
        self.boxes.clear();
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
) -> Result<LayoutBox> {
    let (top, bottom) = (d.item_top(item), d.item_bottom(item));
    let ln_p = ln_between(top, bottom);
    let rank = d.item_rank(item);
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
        y0: placed.y(top),
        y1: placed.y(bottom),
    })
}

/// The surprisal of a gap's children, as well as what is known tells, until
/// samples tell more: with q = p and their number known, that of their mean
/// weight, since children sorted by p are nearly alike side by side;
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
    edges: Vec<(f64, f64)>,
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
        let Some(rank) = p.rank(branch) else {
            break;
        };
        let (top, bottom) = (p.edge(rank), p.edge(rank + 1));
        let (top, bottom) = (top.expect("ranked"), bottom.expect("ranked"));
        let ln_p = ln_between(top, bottom);
        let nats_q = tree.nats_q(node, lens, branch, ln_p)?;
        steps.nodes.push(child_hash(node, branch));
        steps.entries.push(ChainEntry {
            branch,
            rank,
            ln_p,
            nats_q,
        });
        steps.edges.push((top, bottom));
        steps.ln_masses.push(steps.ln_masses[i] + ln_p);
        steps.xs.push(steps.xs[i] + nats_q);
    }
    Ok(steps)
}

/// The known edges nearest a point of a node, above and below it: those of
/// the item holding it, or the node's own.
fn around(d: Option<&Distribution>, point: &Point) -> (f64, f64) {
    d.and_then(|d| Some(d.item_edges(d.locate_by(|edge| point.below(edge))?)))
        .filter(|&(top, bottom)| point.within(top, bottom))
        .unwrap_or((0.0, f64::NEG_INFINITY))
}

/// The known child of a node holding a point, and its edges.
fn held(d: Option<&Distribution>, point: &Point) -> Option<(Branch, f64, f64)> {
    let d = d?;
    let item = d.locate_by(|edge| point.below(edge))?;
    let Item::Child { run, at } = item else {
        return None;
    };
    let (top, bottom) = d.item_edges(item);
    point
        .within(top, bottom)
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
    let y = |edge: f64| {
        let (under, ln_mass) = point.to(edge);
        let far = (ln_scale + ln_mass).exp();
        if under {
            cam.a + far
        } else {
            cam.a - far
        }
    };
    let (top, bottom) = around(tree.distribution(nodes[depth], lens.p)?, &point);
    let anchor = Placed::new(
        (top, y(top)),
        (bottom, y(bottom)),
        ln_masses[depth],
        ln_scale,
        xs[depth],
    );
    let mut placed = vec![anchor; nodes.len()];
    let mut spans = vec![(y(0.0), y(f64::NEG_INFINITY)); nodes.len()];
    for i in (0..depth).rev() {
        let (top, bottom) = edges[i];
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
    /// A child shorter than this, in view heights, joins a band.
    merge: f64,
    frame: Frame,
}

impl Layout<'_> {
    fn push(&mut self, b: LayoutBox) {
        if b.x1 > self.cam.x && b.x0 < self.right && b.y1 > 0.0 && b.y0 < 1.0 {
            self.frame.boxes.push(b);
        }
    }

    /// Lay out the node's children in view, all but the chain child at `skip`.
    fn lay(&mut self, node: NodeId, placed: Placed, skip: Option<Rank>) -> Result<()> {
        let tree = self.tree;
        self.frame.placed.insert(node, placed);
        let Some(d) = tree.distribution(node, self.lens.p)? else {
            return Ok(());
        };
        // From the first item that ends in the view to the last that starts
        // in it, told by view height: a view too small for any float of the
        // node's mass to tell its top from its bottom still has both.
        let mut item = d.locate_by(|edge| placed.y(edge) <= 0.0);
        while let Some(found) = item {
            let (y0, y1) = (placed.y(d.item_top(found)), placed.y(d.item_bottom(found)));
            if 1.0 <= y0 {
                break;
            }
            let chain = matches!(found, Item::Child { .. }) && skip == Some(d.item_rank(found));
            item = if chain {
                d.next(found)
            } else if self.merge <= y1 - y0 {
                match found {
                    Item::Child { .. } => self.child(node, &placed, d, found, (y0, y1))?,
                    Item::Gap { .. } => {
                        self.push(item_box(self.tree, self.lens, node, &placed, d, found)?)
                    }
                }
                d.next(found)
            } else {
                self.band(node, &placed, d, found, skip)?
            };
        }
        Ok(())
    }

    /// A child spanning `y0..y1`, in a node whose view spans the edges `view`.
    fn child(
        &mut self,
        node: NodeId,
        placed: &Placed,
        d: &Distribution,
        item: Item,
        (y0, y1): (f64, f64),
    ) -> Result<()> {
        let tree = self.tree;
        let mut b = item_box(tree, self.lens, node, placed, d, item)?;
        // The band test's edges, so neighbours meet exactly.
        (b.y0, b.y1) = (y0, y1);
        let branch = b.branch.expect("a child has a branch");
        let id = child_hash(node, branch);
        let (x1, kind) = (b.x1, b.kind);
        let (top, bottom) = (d.item_top(item), d.item_bottom(item));
        self.push(b);
        if self.right <= x1 || kind != BoxKind::Node {
            return Ok(());
        }
        self.lay(
            id,
            placed.child(top, bottom, ln_between(top, bottom), x1),
            None,
        )
    }

    /// A band about `merge` tall from `start`: thin children and the gaps
    /// between them, drawn as one silhouette whose width is estimated from
    /// its known children, Σ p (−ln q) / Σ p. The item after the band, if
    /// any.
    fn band(
        &mut self,
        node: NodeId,
        placed: &Placed,
        d: &Distribution,
        start: Item,
        skip: Option<Rank>,
    ) -> Result<Option<Item>> {
        let height = |item: Item| placed.y(d.item_bottom(item)) - placed.y(d.item_top(item));
        let target = placed.ln_below(placed.y(d.item_top(start)) + self.merge);
        let (mut ln_known, mut ln_area) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut ln_ratio = 0.0;
        let mut count = 0usize;
        let mut last = start;
        let mut item = Some(start);
        while let Some(found) = item {
            let thick = self.merge <= height(found);
            if found != start && (thick || Some(d.item_rank(found)) == skip) {
                break;
            }
            match found {
                Item::Child { run, at } => {
                    let r = &d.runs()[run];
                    // Through the child about `merge` below the band's top,
                    // short of the chain child.
                    let mut end = r.holding(target).clamp(at, r.len() - 1);
                    if let Some(skip) = skip.filter(|&skip| r.rank() + (at as Rank) < skip) {
                        end = end
                            .min((skip - r.rank()).saturating_sub(1) as usize)
                            .max(at);
                    }
                    let ln_mass = ln_between(r.edge(at), r.edge(end + 1));
                    if self.lens.p == self.lens.q {
                        ln_known = ln_add(ln_known, ln_mass);
                        ln_area = ln_add(ln_area, r.ln_surprisal(at, end + 1));
                        count += end + 1 - at;
                    } else {
                        let n = end + 1 - at;
                        let (mut mass, mut area, mut ratio) = (0.0, 0.0, 0.0);
                        for sample in 0..SAMPLES.min(n) {
                            let i = at + sample * n / SAMPLES.min(n);
                            let nats = self.tree.nats_q(node, self.lens, r.branch(i), r.ln_p(i))?;
                            if nats.is_finite() {
                                let p = (r.ln_p(i) - r.ln_p(at)).exp();
                                (mass, area) = (mass + p, area + p * nats);
                                ratio += p * (r.ln_p(i) + nats);
                                count += 1;
                            }
                        }
                        if 0.0 < mass {
                            ln_known = ln_add(ln_known, ln_mass);
                            ln_ratio += (ratio / mass - ln_ratio) * (ln_mass - ln_known).exp();
                            ln_area = ln_add(ln_area, ln_mass + (area / mass).ln());
                        }
                    }
                    last = Item::Child { run, at: end };
                    item = d.next(last);
                    if r.edge(end + 1) <= target {
                        break;
                    }
                }
                Item::Gap { .. } => {
                    last = found;
                    item = d.next(found);
                    if d.item_bottom(found) <= target {
                        break;
                    }
                }
            }
        }
        let (top, bottom) = (d.item_top(start), d.item_bottom(last));
        let (y0, y1) = (placed.y(top), placed.y(bottom));
        if ln_known == f64::NEG_INFINITY {
            // Nothing known to draw.
            return Ok(item);
        }
        let nats = (ln_area - ln_known).exp();
        let ln_p = ln_between(top, bottom);
        self.push(LayoutBox {
            node,
            rank: d.item_rank(start),
            count: count as u32,
            kind: BoxKind::Merged,
            branch: None,
            // q already represents a typical child, not the whole band.
            // Use the same p-weighted geometric mean for p so ln(p/q)
            // is the children's mean log ratio, independent of grouping.
            p: (ln_ratio - nats).exp(),
            q: (-nats).exp(),
            ln_mass: placed.ln_mass + ln_p,
            x0: placed.x,
            x1: placed.x + nats,
            y0,
            y1,
        });
        Ok(item)
    }
}

pub fn layout(
    tree: &Tree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    merge_px: f64,
) -> Result<Frame> {
    layout_into(tree, lens, cam, view, merge_px, Frame::default())
}

pub(crate) fn layout_into(
    tree: &Tree,
    lens: Lens,
    cam: &Camera,
    view: Viewport,
    merge_px: f64,
    mut frame: Frame,
) -> Result<Frame> {
    frame.clear();
    let Chain {
        nodes,
        entries,
        placed,
        spans,
    } = place_chain(tree, lens, cam)?;
    for (&node, &placed) in nodes.iter().zip(&placed) {
        frame.placed.insert(node, placed);
    }
    (frame.origin_top, frame.origin_bottom) = spans[0];
    let mut context = Layout {
        tree,
        lens,
        cam,
        right: cam.x + view.width / cam.k,
        merge: merge_px / view.height,
        frame,
    };
    let depth = nodes.len() - 1;
    context.lay(nodes[depth], placed[depth], None)?;
    for i in (0..depth).rev() {
        let (own, child, entry) = (placed[i], placed[i + 1], &entries[i]);
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
        context.lay(nodes[i], own, Some(entry.rank))?;
        // Covering the view with its own box left of it, this node hides
        // every remaining ancestor's other children.
        let (top, bottom) = spans[i];
        if top <= 0.0 && bottom >= 1.0 && own.x <= cam.x {
            break;
        }
    }
    Ok(context.frame)
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
            let start =
                Point::off(0.0, true, from - top, -at.ln_scale).unwrap_or_else(|| Point::new(&[]));
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
        _ if y1.abs() < y0.abs() => {
            Point::off(f64::NEG_INFINITY, y1 <= 0.0, y1.abs(), -placed.ln_scale)
        }
        _ => Point::off(0.0, true, y0.abs(), -placed.ln_scale),
    }
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
    let covers = |point: &Point, ln_scale: f64, top: f64, bottom: f64| {
        let ((_, above), (_, under)) = (point.to(top), point.to(bottom));
        cam.a - (ln_scale + above).exp() <= FIT && 1.0 - FIT <= cam.a + (ln_scale + under).exp()
    };
    while let Some(&(top, bottom)) = path.len().checked_sub(1).map(|i| &steps.edges[i]) {
        let ln_scale = steps.ln_masses[path.len()] + cam.x;
        if covers(&point, ln_scale, 0.0, f64::NEG_INFINITY) {
            break;
        }
        point = point.out_of_child(top, bottom);
        path.pop();
    }
    let mut node = steps.nodes[path.len()];
    let mut ln_scale = steps.ln_masses[path.len()] + cam.x;
    while let Some((branch, top, bottom)) = held(tree.distribution(node, lens.p)?, &point) {
        if !covers(&point, ln_scale, top, bottom) {
            break;
        }
        ln_scale += ln_between(top, bottom);
        point = point.into_child(top, bottom, precision(ln_scale));
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
/// `(x_px, y_px)`: a child, however thin and merged, found by walking down
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
    let (x, y) = (cam.x + x_px / cam.k, y_px / view.height);
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
        let Some(item) = d.locate_by(|edge| placed.y(edge) <= y) else {
            return Ok(Some(within.map_or(origin, |b| PickResult {
                r#box: b,
                exact: true,
            })));
        };
        let item = band_child(frame, node, placed, d, item, y, y + pixel_px / view.height);
        let b = item_box(tree, lens, node, placed, d, item)?;
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

/// Bands draw thin children and the gaps between them as one silhouette,
/// so on a pixel of one, from `y` to `bottom`, the known child the pixel
/// holds is selected rather than the gap at its corner, whichever touching
/// band draws it. A gap drawn as its own box keeps its pixels: the child
/// after it starts on the next one.
fn band_child(
    frame: &Frame,
    node: NodeId,
    placed: &Placed,
    d: &Distribution,
    item: Item,
    y: f64,
    bottom: f64,
) -> Item {
    let banded = |y: f64| {
        frame
            .boxes
            .iter()
            .any(|b| b.node == node && b.kind == BoxKind::Merged && b.y0 <= y && y < b.y1)
    };
    if !matches!(item, Item::Gap { .. }) || !banded(y) {
        return item;
    }
    match d.next(item) {
        Some(next @ Item::Child { .. }) => {
            let top = placed.y(d.item_top(next));
            if top < bottom && banded(top) {
                next
            } else {
                item
            }
        }
        _ => item,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{Distribution, Run};
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
        let mut tree = Tree::new(1);
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

        let mut tree = Tree::new(2);
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

    #[test]
    fn thin_children_merge_into_bands_that_keep_their_area_and_pick_apart() {
        let weights_ = [0.5, 0.488, 0.006, 0.004, 0.002];
        let (tree, _) = chain_tree(&weights_, &[]);
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&tree, LENS, &camera, VIEW, 10.0).unwrap();
        let band = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Merged)
            .expect("the light children merge");
        let area: f64 = weights_[2..].iter().map(|w| -w * w.ln()).sum();
        assert!((band.ln_mass.exp() * (band.x1 - band.x0) - area).abs() < 1e-12);
        assert_eq!(band.count, 3);
        let hit = pick(
            &tree,
            LENS,
            &frame,
            &camera,
            VIEW,
            100.0,
            0.991 * VIEW.height,
            1.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(hit.r#box.branch, Some(2));
        assert!(hit.exact);
    }

    #[test]
    fn bands_color_by_their_childrens_mean_log_ratio() {
        let p = [0.5, 0.488, 0.006, 0.004, 0.002];
        let q = [0.5, 0.48, 0.004, 0.008, 0.008];
        let mut tree = Tree::new(2);
        tree.set(ROOT, 0, weights(&p)).unwrap();
        tree.set(ROOT, 1, weights(&q)).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let camera = cam(ROOT, 0.0, top(&[]), 0.0);
        let frame = layout(&tree, lens, &camera, VIEW, 10.0).unwrap();
        let band = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Merged)
            .unwrap();
        let expected = p[2..]
            .iter()
            .zip(&q[2..])
            .map(|(p, q)| p * (p / q).ln())
            .sum::<f64>()
            / p[2..].iter().sum::<f64>();
        near((band.p / band.q).ln(), expected);

        // Identical measures must stay neutral at every grouping size.
        tree.set(ROOT, 1, weights(&p)).unwrap();
        for merge in [1.0, 10.0, 100.0] {
            let frame = layout(&tree, lens, &camera, VIEW, merge).unwrap();
            for band in frame.boxes.iter().filter(|b| b.kind == BoxKind::Merged) {
                near((band.p / band.q).ln(), 0.0);
            }
        }
    }

    #[test]
    fn touching_bands_share_their_edge_exactly_across_runs() {
        // One heavy child, then thin known children alternating with
        // one-child gaps: every band boundary is a run's or a gap's edge.
        // The renderer joins touching bands by comparing edges exactly.
        let w = 1.25e-5;
        let mut runs = vec![Run::new(0, vec![0], vec![0.0, 0.01f64.ln()]).unwrap()];
        for i in 0..400u32 {
            let below = 0.01 - 2.0 * f64::from(i) * w;
            runs.push(
                Run::new(
                    1 + 2 * i,
                    vec![1 + 2 * i],
                    vec![below.ln(), (below - w).ln()],
                )
                .unwrap(),
            );
        }
        let mut tree = Tree::new(1);
        tree.insert(ROOT, 0, runs).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let bands: Vec<_> = frame
            .boxes
            .iter()
            .filter(|b| b.kind == BoxKind::Merged)
            .collect();
        assert!(2 < bands.len());
        for pair in bands.windows(2) {
            assert_eq!(pair[1].y0, pair[0].y1);
        }
    }

    #[test]
    fn a_band_pixel_selects_a_known_child_it_holds_over_a_gap() {
        // Known children alternate with one-child gaps, about 40 a pixel.
        let w = 1.25e-5;
        let mut runs = vec![Run::new(0, vec![0], vec![0.0, 0.01f64.ln()]).unwrap()];
        for i in 0..400u32 {
            let below = 0.01 - 2.0 * f64::from(i) * w;
            runs.push(
                Run::new(
                    1 + 2 * i,
                    vec![1 + 2 * i],
                    vec![below.ln(), (below - w).ln()],
                )
                .unwrap(),
            );
        }
        let mut tree = Tree::new(1);
        tree.insert(ROOT, 0, runs).unwrap();
        // Off the grid, so pixel corners fall inside items.
        let camera = cam(ROOT, 0.0, top(&[]), 0.37 / VIEW.height);
        let frame = layout(&tree, LENS, &camera, VIEW, 1.0).unwrap();
        let selected = |y_px: f64, pixel_px: f64| {
            let hit = pick(&tree, LENS, &frame, &camera, VIEW, 100.0, y_px, pixel_px);
            hit.unwrap().unwrap().r#box.branch
        };
        let rows = frame
            .boxes
            .iter()
            .filter(|b| b.kind == BoxKind::Merged)
            .flat_map(|b| (b.y0 * VIEW.height).ceil() as u32..(b.y1 * VIEW.height).ceil() as u32)
            .filter(|&row| f64::from(row) < VIEW.height);
        let mut corners_in_gaps = 0;
        for row in rows {
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
            Run::new(0, vec![0], vec![0.0, 0.5f64.ln()]).unwrap(),
            Run::new(5, vec![5], vec![below.ln(), (below - 0.1).ln()]).unwrap(),
        ];
        let mut tree = Tree::new(1);
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
        // Three children 2^-3000 of the root, below a heavy one: no float of
        // a cdf tells them apart, but ln-below edges do.
        let tiny = -3000.0 * LN_2;
        let mut tree = Tree::new(1);
        let ln_below = vec![
            0.0,
            tiny + 3f64.ln(),
            tiny + 2f64.ln(),
            tiny,
            f64::NEG_INFINITY,
        ];
        tree.set(
            ROOT,
            0,
            Distribution::new(Run::new(0, vec![0, 1, 2, 3], ln_below).unwrap()),
        )
        .unwrap();
        // The view is the tail itself: x = 3000 bits is where children
        // 2^-3000 of the root fill it.
        // The top of the first of them, 3 · 2^-3000 above the root's bottom.
        let mut digits = vec![0xff; 374];
        digits.extend([0xfd]);
        let camera = cam(ROOT, -tiny - 3f64.ln(), at(&[], &digits), 0.0);
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
    fn gaps_draw_at_their_mean_weight_and_bands_from_their_samples() {
        let mut tree = Tree::new(1);
        // The two heaviest children of the root, and the node's end at rank
        // 1000: a quarter of its mass in 998 unknown children.
        let run = Run::new(0, vec![4, 7], vec![0.0, 0.5f64.ln(), 0.25f64.ln()]).unwrap();
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
        let mut uncounted = Tree::new(1);
        let run = Run::new(0, vec![4, 7], vec![0.0, 0.5f64.ln(), 0.25f64.ln()]).unwrap();
        uncounted.insert(ROOT, 0, vec![run]).unwrap();
        let frame_ = layout(&uncounted, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let gap_ = frame_
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .unwrap();
        near(gap_.x1 - gap_.x0, 4f64.ln());
        // An answer of four samples splits the gap into thin gaps: the bands
        // across them draw a silhouette from the samples.
        let samples: Vec<_> = (0..4)
            .map(|i| {
                let top = (0.25 - 0.0625 * f64::from(i)).ln();
                Run::new(
                    2 + 100 * i,
                    vec![100 + i],
                    vec![top, (top.exp() - 1e-4).ln()],
                )
                .unwrap()
            })
            .collect();
        tree.insert(ROOT, 0, samples).unwrap();
        let frame = layout(&tree, LENS, &cam(ROOT, 0.0, top(&[]), 0.0), VIEW, 1.0).unwrap();
        let bands = frame
            .boxes
            .iter()
            .filter(|b| b.kind == BoxKind::Merged)
            .count();
        assert!(0 < bands);
        let width = -(1e-4f64).ln();
        for band in frame.boxes.iter().filter(|b| b.kind == BoxKind::Merged) {
            near(band.x1 - band.x0, width);
        }
    }

    #[test]
    fn a_view_no_float_of_a_node_resolves_is_told_exactly() {
        // A child 2^-80 of it in view, a quarter of the way down it: its
        // masses below the view's top and bottom are one float.
        let mut tree = Tree::new(1);
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

        let mut tree = Tree::new(2);
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
        let mut tree = Tree::new(1);
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

        let empty = Tree::new(1);
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
        tree.set(ended, 0, Distribution::from_weights(&[]).unwrap())
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
