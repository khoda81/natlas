//! Anchored geometry. A node's children split its height by p, in p's order,
//! and reach −ln q to the right. A frame lays out what a camera sees, says
//! what it lacks, and answers picks.
//!
//! Positions never come from absolute coordinates: each laid-out node keeps
//! two of its own edges placed exactly in the view and measures everything
//! from the nearer one, so a branch 2^-4096 of its parent sits as exactly
//! beside the anchor as a heavy one.

use std::collections::HashMap;
use std::f64::consts::LN_2;

use serde::{Deserialize, Serialize};

use crate::distribution::{ln_add, ln_between, Distribution, Item, Rank, Run, Window};
use crate::tree::{child_hash, node_id, Branch, Lens, NodeId, Tree};
use crate::Result;

/// Shorter than this many pixels, a region asks a source for nothing.
const MIN_PX: f64 = 2.0;
/// Children down to 2^-HEADROOM of a pixel are asked for, so a little zoom
/// needs nothing new.
const HEADROOM: u32 = 4;
const MAX_HORIZON: u32 = 4096;
/// Children sampled for a merged band's width when q is another measure.
const SAMPLES: usize = 4;

/// The top edge of a node's child at `rank`; rank 0 is the node's top.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Edge {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub rank: Rank,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    /// The node the view is measured from, standing as the root does: x is
    /// q-surprisal from its children's left edge and the view spans e^-x of
    /// its interval. It bounds the view; nothing above it is needed.
    #[serde(with = "node_id")]
    pub origin: NodeId,
    pub x: f64,
    pub k: f64,
    /// A known edge on a node at or below the origin...
    pub anchor: Edge,
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
    pub p: f64,
    /// NaN while unknown.
    pub q: f64,
    pub ln_mass: f64,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

/// What a frame lacks, as the Expand request that fills it: the window of
/// `node` in view, as ln of the mass below its edges, listed down to
/// `horizon`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Miss {
    #[serde(with = "node_id")]
    pub node: NodeId,
    /// The node whose children are lacking: `node`, or a child it holds.
    #[serde(with = "node_id")]
    pub target: NodeId,
    pub measure: usize,
    /// The target's lacking ranks, from `from` to `to` or to its end: while
    /// they stay the same, asking again finds nothing new.
    pub from: Rank,
    pub to: Option<Rank>,
    pub ln_top: f64,
    pub ln_bottom: f64,
    pub horizon: u32,
    /// Visible pixels; larger first.
    pub priority: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct EdgePosition {
    pub edge: Edge,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cover {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Extent {
    pub top: f64,
    pub bottom: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnchorPosition {
    pub anchor: Edge,
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
    /// ln of the node's p mass below the origin.
    ln_mass: f64,
    /// ln of view heights per unit of the node's mass.
    ln_scale: f64,
    /// Where the node's children start, in nats right of the origin's.
    x: f64,
    /// Below the origin.
    depth: usize,
}

impl Placed {
    /// The view height of an edge, measured from the placed edge nearer to it.
    fn y(&self, ln_below: f64) -> f64 {
        let (edge, y) = if ln_below >= self.top.0 {
            self.top
        } else if ln_below <= self.bottom.0
            || ln_between(self.top.0, ln_below) > ln_between(ln_below, self.bottom.0)
        {
            self.bottom
        } else {
            self.top
        };
        if ln_below <= edge {
            y + (self.ln_scale + ln_between(edge, ln_below)).exp()
        } else {
            y - (self.ln_scale + ln_between(ln_below, edge)).exp()
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
        Placed {
            top: (0.0, self.y(top)),
            bottom: (f64::NEG_INFINITY, self.y(bottom)),
            ln_mass: self.ln_mass + ln_p,
            ln_scale: self.ln_scale + ln_p,
            x,
            depth: self.depth + 1,
        }
    }
}

#[derive(Debug, Default)]
pub struct Frame {
    pub boxes: Vec<LayoutBox>,
    /// Every node laid out, for picking.
    pub placed: HashMap<NodeId, Placed>,
    pub misses: Vec<Miss>,
    /// Known edges in view, for re-anchoring.
    pub edges: Vec<EdgePosition>,
    pub origin_top: f64,
    pub origin_bottom: f64,
    /// Deepest node covering the view whose children start at or left of it.
    pub cover: Option<Cover>,
}

impl Frame {
    /// Empty, keeping the allocations for the next view.
    pub(crate) fn clear(&mut self) {
        self.boxes.clear();
        self.placed.clear();
        self.misses.clear();
        self.edges.clear();
        self.origin_top = 0.0;
        self.origin_bottom = 0.0;
        self.cover = None;
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
        Item::Gap { .. } => (BoxKind::Gap, None, f64::INFINITY),
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

/// The horizon that lists a window `px` tall down to 2^-HEADROOM of a pixel.
fn horizon(px: f64) -> u32 {
    (px.log2().ceil().max(0.0) as u32 + HEADROOM).min(MAX_HORIZON)
}

struct ChainEntry {
    rank: Rank,
    ln_p: f64,
    nats_q: f64,
}

/// The nodes from the origin to the anchor, each one's entry in its parent,
/// and where each is placed: the anchor at its edge, every ancestor at its
/// chain child's edges.
struct Chain {
    nodes: Vec<NodeId>,
    entries: Vec<ChainEntry>,
    placed: Vec<Placed>,
}

fn place_chain(tree: &Tree, lens: Lens, cam: Camera) -> Result<Chain> {
    let lambda = -cam.x;
    let nodes = tree.chain(cam.origin, cam.anchor.node)?;
    let mut entries = Vec::with_capacity(nodes.len());
    let mut edges = Vec::with_capacity(nodes.len());
    let (mut ln_masses, mut xs) = (vec![0.0], vec![0.0]);
    for pair in nodes.windows(2) {
        let branch = tree.branch(pair[1])?.expect("a child has a branch");
        let p = tree.distribution(pair[0], lens.p)?;
        let rank = p.and_then(|p| p.rank(branch)).ok_or_else(|| {
            format!(
                "node {:#018x} is not placed under measure {}",
                pair[1], lens.p
            )
        })?;
        let p = p.expect("a ranked branch has a distribution");
        let (top, bottom) = (p.edge(rank), p.edge(rank + 1));
        let (top, bottom) = (top.expect("ranked"), bottom.expect("ranked"));
        let ln_p = ln_between(top, bottom);
        let nats_q = tree.nats_q(pair[0], lens, branch, ln_p)?;
        ln_masses.push(ln_masses.last().expect("origin") + ln_p);
        xs.push(xs.last().expect("origin") + nats_q);
        entries.push(ChainEntry { rank, ln_p, nats_q });
        edges.push((top, bottom));
    }
    let depth = nodes.len() - 1;
    let anchor = match cam.anchor.rank {
        0 => Some(0.0),
        rank => tree
            .distribution(cam.anchor.node, lens.p)?
            .and_then(|d| d.edge(rank)),
    }
    .ok_or("the anchor's edge is unknown")?;
    let mut placed = vec![
        Placed {
            top: (anchor, cam.a),
            bottom: (anchor, cam.a),
            ln_mass: ln_masses[depth],
            ln_scale: ln_masses[depth] - lambda,
            x: xs[depth],
            depth,
        };
        nodes.len()
    ];
    for i in (0..depth).rev() {
        let (top, bottom) = edges[i];
        let (y0, y1) = placed[i + 1].span();
        placed[i] = Placed {
            top: (top, y0),
            bottom: (bottom, y1),
            ln_mass: ln_masses[i],
            ln_scale: ln_masses[i] - lambda,
            x: xs[i],
            depth: i,
        };
    }
    Ok(Chain {
        nodes,
        entries,
        placed,
    })
}

struct Layout<'a> {
    tree: &'a Tree,
    lens: Lens,
    cam: Camera,
    view: Viewport,
    right: f64,
    /// A child shorter than this, in view heights, joins a band.
    merge: f64,
    frame: Frame,
    cover_depth: Option<usize>,
}

impl Layout<'_> {
    fn push(&mut self, b: LayoutBox) {
        if b.x1 > self.cam.x && b.x0 < self.right && b.y1 > 0.0 && b.y0 < 1.0 {
            self.frame.boxes.push(b);
        }
    }

    fn edge(&mut self, node: NodeId, rank: Rank, y: f64) {
        if (0.0..=1.0).contains(&y) {
            self.frame.edges.push(EdgePosition {
                edge: Edge { node, rank },
                y,
            });
        }
    }

    fn cover(&mut self, node: NodeId, y0: f64, y1: f64, x: f64, depth: usize) {
        if y0 <= 1e-12
            && y1 >= 1.0 - 1e-12
            && x <= self.cam.x + 1e-12
            && self.cover_depth.is_none_or(|deepest| depth > deepest)
        {
            self.cover_depth = Some(depth);
            self.frame.cover = Some(Cover {
                node,
                y: y0.min(0.0),
            });
        }
    }

    /// Visible pixels between two view heights.
    fn px(&self, y0: f64, y1: f64) -> f64 {
        (y1.min(1.0) - y0.max(0.0)).max(0.0) * self.view.height
    }

    /// Ask for a miss whose window is not empty.
    fn want(&mut self, miss: Miss) {
        if Window::new(miss.ln_top, miss.ln_bottom).is_ok() {
            self.frame.misses.push(miss);
        }
    }

    /// Lay out the node's children in view, all but the chain child at `skip`.
    fn lay(&mut self, node: NodeId, placed: Placed, skip: Option<Rank>) -> Result<()> {
        let tree = self.tree;
        self.frame.placed.insert(node, placed);
        let (top, bottom) = (placed.ln_below(0.0), placed.ln_below(1.0));
        let Some(d) = tree.distribution(node, self.lens.p)? else {
            let (y0, y1) = placed.span();
            let px = self.px(y0, y1);
            if px >= MIN_PX {
                self.want(Miss {
                    node,
                    target: node,
                    measure: self.lens.p,
                    from: 0,
                    to: None,
                    ln_top: top,
                    ln_bottom: bottom,
                    horizon: horizon(px),
                    priority: px,
                });
            }
            return Ok(());
        };
        self.want_q(node, &placed)?;
        let mut item = d.locate(top);
        while let Some(found) = item {
            if d.item_top(found) <= bottom {
                break;
            }
            item = match found {
                Item::Gap { before } => {
                    self.gap(node, &placed, d, before, top, bottom)?;
                    d.next(found)
                }
                Item::Child { run, at } => {
                    let r = &d.runs()[run];
                    let rank = r.rank() + at as Rank;
                    let (y0, y1) = (placed.y(r.edge(at)), placed.y(r.edge(at + 1)));
                    if skip == Some(rank) {
                        d.next(found)
                    } else if y1 - y0 >= self.merge {
                        self.child(node, &placed, d, found, (y0, y1), (top, bottom))?;
                        d.next(found)
                    } else {
                        // A band through the child about `merge` below, short
                        // of the chain child.
                        let mut end = r
                            .holding(placed.ln_below(y0 + self.merge))
                            .clamp(at, r.len() - 1);
                        if let Some(skip) = skip.filter(|&skip| skip > rank) {
                            end = end.min((skip - r.rank()) as usize - 1);
                        }
                        self.band(node, &placed, r, at, end)?;
                        d.next(Item::Child { run, at: end })
                    }
                }
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
        view: (f64, f64),
    ) -> Result<()> {
        let tree = self.tree;
        let mut b = item_box(tree, self.lens, node, placed, d, item)?;
        // The band test's edges, so neighbours meet exactly.
        (b.y0, b.y1) = (y0, y1);
        let branch = b.branch.expect("a child has a branch");
        let id = child_hash(node, branch);
        let (rank, x1, kind) = (b.rank, b.x1, b.kind);
        let (top, bottom) = (d.item_top(item), d.item_bottom(item));
        self.edge(node, rank, y0);
        self.edge(node, rank + 1, y1);
        self.push(b);
        if tree.contains(id) {
            self.cover(id, y0, y1, x1, placed.depth + 1);
        }
        if x1 >= self.right {
            return Ok(());
        }
        let child = placed.child(top, bottom, ln_between(top, bottom), x1);
        if kind == BoxKind::Node {
            return self.lay(id, child, None);
        }
        // The part in view, as a window on the parent, which the source has
        // already reached: it descends into the child and lists it there.
        let px = self.px(y0, y1);
        if px >= MIN_PX {
            self.want(Miss {
                node,
                target: id,
                measure: self.lens.p,
                from: 0,
                to: None,
                ln_top: top.min(view.0),
                ln_bottom: bottom.max(view.1),
                horizon: horizon(px),
                priority: px,
            });
        }
        Ok(())
    }

    fn gap(
        &mut self,
        node: NodeId,
        placed: &Placed,
        d: &Distribution,
        before: usize,
        top: f64,
        bottom: f64,
    ) -> Result<()> {
        let item = Item::Gap { before };
        let b = item_box(self.tree, self.lens, node, placed, d, item)?;
        let (rank, y0, y1) = (b.rank, b.y0, b.y1);
        self.edge(node, rank, y0);
        if let Some(end) = d.item_end(item) {
            self.edge(node, end, y1);
        }
        self.push(b);
        // Ask only where a child could show: the heaviest a gap can hold is
        // as heavy as the child above it.
        let px = self.px(y0, y1);
        let ceiling = (placed.ln_scale + d.gap_ceiling(before)).exp() * self.view.height;
        if px >= MIN_PX && ceiling >= 0.5f64.powi(HEADROOM as i32) {
            self.want(Miss {
                node,
                target: node,
                measure: self.lens.p,
                from: rank,
                to: d.item_end(item),
                ln_top: d.item_top(item).min(top),
                ln_bottom: d.item_bottom(item).max(bottom),
                horizon: horizon(px),
                priority: px,
            });
        }
        Ok(())
    }

    /// Children [from, to] of a run as one band. Its width keeps its area:
    /// Σ p (−ln q) / Σ p, exact when q is p, sampled otherwise.
    fn band(
        &mut self,
        node: NodeId,
        placed: &Placed,
        run: &Run,
        from: usize,
        to: usize,
    ) -> Result<()> {
        let (top, bottom) = (run.edge(from), run.edge(to + 1));
        let ln_p = ln_between(top, bottom);
        let nats = if self.lens.p == self.lens.q {
            (run.ln_surprisal(from, to + 1) - ln_p).exp()
        } else {
            let count = to - from + 1;
            let (mut mass, mut area) = (0.0, 0.0);
            for sample in 0..SAMPLES.min(count) {
                let at = from + sample * count / SAMPLES.min(count);
                let p = run.ln_p(at).exp();
                mass += p;
                area += p * self
                    .tree
                    .nats_q(node, self.lens, run.branch(at), run.ln_p(at))?;
            }
            area / mass
        };
        let rank = run.rank() + from as Rank;
        let (y0, y1) = (placed.y(top), placed.y(bottom));
        self.edge(node, rank, y0);
        self.edge(node, run.rank() + to as Rank + 1, y1);
        self.push(LayoutBox {
            node,
            rank,
            count: (to - from + 1) as u32,
            kind: BoxKind::Merged,
            branch: None,
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
        });
        Ok(())
    }

    /// With q another measure, the node's children need q down to the view's
    /// right edge: every one q leaves out is wider than that.
    fn want_q(&mut self, node: NodeId, placed: &Placed) -> Result<()> {
        let bits = (self.right - placed.x) / LN_2;
        if self.lens.p == self.lens.q || bits.is_nan() || bits <= 0.0 {
            return Ok(());
        }
        let need = (bits.ceil() as u32).min(MAX_HORIZON);
        let q = self.tree.distribution(node, self.lens.q)?;
        if q.is_some_and(|q| q.covers(-f64::from(need) * LN_2)) {
            return Ok(());
        }
        let head = q
            .and_then(|q| q.runs().first())
            .filter(|run| run.rank() == 0);
        let (y0, y1) = placed.span();
        let px = self.px(y0, y1);
        self.want(Miss {
            node,
            target: node,
            measure: self.lens.q,
            from: head.map_or(0, Run::end),
            to: q.and_then(|q| q.runs().get(usize::from(head.is_some())).map(Run::rank)),
            ln_top: 0.0,
            ln_bottom: f64::NEG_INFINITY,
            horizon: (need + HEADROOM).min(MAX_HORIZON),
            priority: px,
        });
        Ok(())
    }
}

pub fn layout(
    tree: &Tree,
    lens: Lens,
    cam: Camera,
    view: Viewport,
    merge_px: f64,
) -> Result<Frame> {
    layout_into(tree, lens, cam, view, merge_px, Frame::default())
}

pub(crate) fn layout_into(
    tree: &Tree,
    lens: Lens,
    cam: Camera,
    view: Viewport,
    merge_px: f64,
    mut frame: Frame,
) -> Result<Frame> {
    frame.clear();
    let Chain {
        nodes,
        entries,
        placed,
    } = place_chain(tree, lens, cam)?;
    for (&node, &placed) in nodes.iter().zip(&placed) {
        frame.placed.insert(node, placed);
    }
    (frame.origin_top, frame.origin_bottom) = placed[0].span();
    let mut context = Layout {
        tree,
        lens,
        cam,
        view,
        right: cam.x + view.width / cam.k,
        merge: merge_px / view.height,
        frame,
        cover_depth: None,
    };
    let depth = nodes.len() - 1;
    let anchor = placed[depth];
    context.edge(cam.anchor.node, cam.anchor.rank, cam.a);
    context.lay(nodes[depth], anchor, None)?;
    let (y0, y1) = anchor.span();
    context.cover(nodes[depth], y0, y1, anchor.x, depth);
    for i in (0..depth).rev() {
        let (own, child, entry) = (placed[i], placed[i + 1], &entries[i]);
        let (y0, y1) = child.span();
        context.edge(nodes[i], entry.rank, y0);
        context.edge(nodes[i], entry.rank + 1, y1);
        context.push(LayoutBox {
            node: nodes[i],
            rank: entry.rank,
            count: 1,
            kind: BoxKind::Node,
            branch: tree.branch(nodes[i + 1])?,
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
        let (top, bottom) = own.span();
        context.cover(nodes[i], top, bottom, own.x, i);
        // Covering the view with its own box left of it, this node hides
        // every remaining ancestor's other children.
        if top <= 0.0 && bottom >= 1.0 && own.x <= cam.x {
            break;
        }
    }
    Ok(context.frame)
}

/// The origin's edges in view heights.
pub fn extent(tree: &Tree, lens: Lens, cam: Camera) -> Result<Extent> {
    let (top, bottom) = place_chain(tree, lens, cam)?.placed[0].span();
    Ok(Extent { top, bottom })
}

/// Keep the view inside the root's interval, returning only the new anchor offset.
pub fn clamp_position(tree: &Tree, lens: Lens, cam: Camera) -> Result<f64> {
    // Only the root's interval is all there is; beyond any other origin lies
    // its parent, unknown until fetched.
    if cam.origin != crate::tree::ROOT {
        return Ok(cam.a);
    }
    let Extent { top, bottom } = extent(tree, lens, cam)?;
    Ok(if top > 0.0 {
        cam.a - top
    } else if bottom < 1.0 {
        cam.a + 1.0 - bottom
    } else {
        cam.a
    })
}

/// The camera's anchor, or the deepest edge above it that p still places:
/// a lens or a replaced distribution can take an edge away.
pub fn repair(tree: &Tree, lens: Lens, cam: Camera) -> Result<Edge> {
    let Ok(chain) = tree.chain(cam.origin, cam.anchor.node) else {
        return Ok(Edge {
            node: cam.origin,
            rank: 0,
        });
    };
    for pair in chain.windows(2) {
        let branch = tree.branch(pair[1])?.expect("a child has a branch");
        let placed = tree
            .distribution(pair[0], lens.p)?
            .is_some_and(|p| p.rank(branch).is_some());
        if !placed {
            return Ok(Edge {
                node: pair[0],
                rank: 0,
            });
        }
    }
    let known = cam.anchor.rank == 0
        || tree
            .distribution(cam.anchor.node, lens.p)?
            .is_some_and(|p| p.edge(cam.anchor.rank).is_some());
    Ok(if known {
        cam.anchor
    } else {
        Edge {
            node: cam.anchor.node,
            rank: 0,
        }
    })
}

/// Re-anchor at the in-view edge nearest the view's center.
pub fn reanchor(frame: &Frame) -> Option<AnchorPosition> {
    frame
        .edges
        .iter()
        .min_by(|a, b| (a.y - 0.5).abs().total_cmp(&(b.y - 0.5).abs()))
        .map(|best| AnchorPosition {
            anchor: best.edge,
            a: best.y,
        })
}

/// The box under a screen point: a child, however thin and merged, found by
/// walking down from the origin through the nodes the frame laid out.
pub fn pick(
    tree: &Tree,
    lens: Lens,
    frame: &Frame,
    cam: Camera,
    view: Viewport,
    x_px: f64,
    y_px: f64,
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
    loop {
        let (Some(placed), Some(d)) = (frame.placed.get(&node), tree.distribution(node, lens.p)?)
        else {
            return Ok(Some(origin));
        };
        let Some(item) = d.locate(placed.ln_below(y)) else {
            return Ok(Some(origin));
        };
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
        node = child;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{Distribution, Run};
    use crate::tree::ROOT;

    const VIEW: Viewport = Viewport {
        width: 1000.0,
        height: 800.0,
    };
    const LENS: Lens = Lens { p: 0, q: 0 };

    fn weights(weights: &[f64]) -> Distribution {
        Distribution::from_weights(&weights.iter().map(|w| w.ln()).collect::<Vec<_>>()).unwrap()
    }

    fn cam(origin: NodeId, x: f64, anchor: Edge, a: f64) -> Camera {
        Camera {
            origin,
            x,
            k: 100.0,
            anchor,
            a,
        }
    }

    fn top(node: NodeId) -> Edge {
        Edge { node, rank: 0 }
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
        let frame = layout(&tree, LENS, cam(ROOT, 0.0, top(ROOT), 0.0), VIEW, 1.0).unwrap();
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
        let frame = layout(&tree, lens, cam(ROOT, 0.0, top(ROOT), 0.0), VIEW, 1.0).unwrap();
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
        let camera = cam(ROOT, 0.0, top(ROOT), 0.0);
        let frame = layout(&tree, LENS, camera, VIEW, 10.0).unwrap();
        let band = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Merged)
            .expect("the light children merge");
        let area: f64 = weights_[2..].iter().map(|w| -w * w.ln()).sum();
        assert!((band.p * (band.x1 - band.x0) - area).abs() < 1e-12);
        assert_eq!(band.count, 3);
        let hit = pick(
            &tree,
            LENS,
            &frame,
            camera,
            VIEW,
            100.0,
            0.991 * VIEW.height,
        )
        .unwrap()
        .unwrap();
        assert_eq!(hit.r#box.branch, Some(2));
        assert!(hit.exact);
    }

    #[test]
    fn deep_anchors_stay_exact_beyond_absolute_float_precision() {
        // 80 halvings put the deepest node 2^-80 of the root tall.
        let path: Vec<Branch> = (0..80).map(|i| i % 2).collect();
        let (tree, deepest) = chain_tree(&[0.5, 0.5], &path);
        let camera = cam(
            ROOT,
            80.0 * LN_2,
            Edge {
                node: deepest,
                rank: 1,
            },
            0.5,
        );
        let frame = layout(&tree, LENS, camera, VIEW, 1.0).unwrap();
        let mut children: Vec<_> = frame.boxes.iter().filter(|b| b.node == deepest).collect();
        children.sort_by_key(|b| b.rank);
        assert_eq!(children.len(), 2);
        near(children[0].y0, 0.0);
        near(children[0].y1, 0.5);
        near(children[1].y1, 1.0);
        // The deepest node fills the view, its children starting at its left edge.
        assert_eq!(frame.cover.map(|c| c.node), Some(deepest));
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
        let camera = cam(
            ROOT,
            -tiny - 3f64.ln(),
            Edge {
                node: ROOT,
                rank: 1,
            },
            0.0,
        );
        let frame = layout(&tree, LENS, camera, VIEW, 1.0).unwrap();
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
    fn a_frame_asks_for_what_it_lacks_and_shows_gaps_reaching_past_the_view() {
        let mut tree = Tree::new(1);
        // Only the two heaviest children of the root are known.
        let run = Run::new(0, vec![4, 7], vec![0.0, 0.5f64.ln(), 0.25f64.ln()]).unwrap();
        tree.insert(ROOT, 0, run).unwrap();
        let frame = layout(&tree, LENS, cam(ROOT, 0.0, top(ROOT), 0.0), VIEW, 1.0).unwrap();
        let gap = frame
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Gap)
            .expect("the unknown quarter is a gap");
        assert_eq!((gap.rank, gap.count, gap.x1), (2, 0, f64::INFINITY));
        near(gap.y0, 0.75);
        let named: Vec<_> = frame
            .misses
            .iter()
            .map(|m| (m.node, m.target, m.from, m.to))
            .collect();
        // The gap from rank 2 to the bottom, and both children through the root.
        assert!(named.contains(&(ROOT, ROOT, 2, None)));
        assert!(named.contains(&(ROOT, child_hash(ROOT, 4), 0, None)));
        assert!(named.contains(&(ROOT, child_hash(ROOT, 7), 0, None)));
        let gap_miss = frame.misses.iter().find(|m| m.from == 2).unwrap();
        near(gap_miss.ln_top, 0.25f64.ln());
        assert_eq!(gap_miss.ln_bottom, f64::NEG_INFINITY);
        // A child is asked for as its part of the parent in view: all of it.
        let child_miss = frame
            .misses
            .iter()
            .find(|m| m.target == child_hash(ROOT, 4))
            .unwrap();
        assert_eq!(child_miss.ln_top, 0.0);
        near(child_miss.ln_bottom, 0.5f64.ln());
    }

    #[test]
    fn reanchoring_moves_nothing_and_repair_falls_back_to_what_p_places() {
        let path: Vec<Branch> = (0..30).map(|i| (i * 7) % 3).collect();
        let (tree, deepest) = chain_tree(&[0.6, 0.3, 0.1], &path);
        let mut camera = cam(ROOT, 12.3, top(deepest), 0.41);
        let before = layout(&tree, LENS, camera, VIEW, 1.0).unwrap();
        let moved = reanchor(&before).unwrap();
        (camera.anchor, camera.a) = (moved.anchor, moved.a);
        let after = layout(&tree, LENS, camera, VIEW, 1.0).unwrap();
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

        let mut tree = Tree::new(2);
        tree.set(ROOT, 0, weights(&[0.5, 0.5])).unwrap();
        tree.set(ROOT, 1, weights(&[1.0])).unwrap();
        let child = tree.child(ROOT, 1).unwrap();
        let lost = cam(ROOT, 1.0, top(child), 0.4);
        assert_eq!(repair(&tree, Lens { p: 1, q: 1 }, lost).unwrap(), top(ROOT));
        assert_eq!(repair(&tree, LENS, lost).unwrap(), top(child));
    }

    #[test]
    fn the_view_clamps_inside_the_root_and_picks_beyond_it_find_the_origin() {
        let (tree, child) = chain_tree(&[0.5, 0.5], &[0]);
        let camera = cam(ROOT, 0.5, top(child), 0.4);
        let a = clamp_position(&tree, LENS, camera).unwrap();
        near(a, 0.0);
        near(
            extent(&tree, LENS, Camera { a, ..camera }).unwrap().top,
            0.0,
        );

        let empty = Tree::new(1);
        let camera = cam(ROOT, 0.0, top(ROOT), 0.0);
        let frame = layout(&empty, LENS, camera, VIEW, 1.0).unwrap();
        assert_eq!(frame.misses.len(), 1);
        let hit = pick(&empty, LENS, &frame, camera, VIEW, 500.0, 400.0)
            .unwrap()
            .unwrap();
        assert_eq!((hit.r#box.node, hit.exact), (ROOT, false));
        assert_eq!(
            pick(&empty, LENS, &frame, camera, VIEW, -1.0, 400.0).unwrap(),
            None
        );
    }
}
