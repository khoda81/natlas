use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use crate::tree::{node_id, Branch, Lens, NodeId, Slots, Tree};
use crate::Result;

pub const MAX_SAFE_SLOT: u64 = 9_007_199_254_740_991;

/// The top edge of a slot; a slot at or past the node's slot count is its bottom.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Boundary {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub slot: u64,
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
    pub anchor: Boundary,
    /// Anchor position in view heights below the top of the view.
    pub a: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BoxKind {
    Node,
    Leaf,
    Tail,
    Merged,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LayoutBox {
    #[serde(with = "node_id")]
    pub node: NodeId,
    /// First slot; a merged box covers `count` consecutive slots.
    pub slot: u64,
    pub count: u64,
    pub kind: BoxKind,
    /// The child's branch; null for a tail or a merged run.
    pub branch: Option<Branch>,
    pub p: f64,
    pub q: f64,
    pub ln_mass: f64,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Walked {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub ln_mass: f64,
    pub x: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Unexpanded {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub branch: Branch,
    pub ln_mass: f64,
    pub x1: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoundaryPosition {
    pub boundary: Boundary,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cover {
    #[serde(with = "node_id")]
    pub node: NodeId,
    pub y: f64,
}

#[derive(Debug, Default)]
pub struct Frame {
    pub boxes: Vec<LayoutBox>,
    /// Known individual branches, including offscreen boxes and merged slots.
    pub known: KnownBranches,
    pub walked: Vec<Walked>,
    pub unexpanded: Vec<Unexpanded>,
    pub boundaries: Vec<BoundaryPosition>,
    pub origin_top: f64,
    pub origin_bottom: f64,
    /// Deepest anchor-chain node covering the view and starting at or left of it.
    pub cover: Option<Cover>,
}

/// Subpixel branches share their immutable slot view and node metadata. Keep
/// the successive accumulated edges: recomputing these from prefix
/// sums would change rounding, grouping boundaries, and precision at depth.
#[derive(Debug)]
struct ThinBranches {
    node: NodeId,
    slots: Arc<Slots>,
    expanded: Arc<HashSet<Branch>>,
    first: usize,
    downward: bool,
    ys: Vec<f64>,
    x: f64,
    ln_mass: f64,
}

impl ThinBranches {
    fn slot(&self, index: usize) -> usize {
        if self.downward {
            self.first + index
        } else {
            self.first - index
        }
    }

    fn get(&self, index: usize) -> LayoutBox {
        let slot = self.slot(index);
        let branch = self.slots.branch(slot);
        LayoutBox {
            node: self.node,
            slot: slot as u64,
            count: 1,
            kind: match branch {
                None => BoxKind::Tail,
                Some(branch) if self.expanded.contains(&branch) => BoxKind::Node,
                _ => BoxKind::Leaf,
            },
            branch,
            p: self.slots.p(slot),
            q: self.slots.q(slot),
            ln_mass: self.ln_mass + self.slots.ln_p(slot),
            x0: self.x,
            x1: self.x + self.slots.nats_q(slot),
            y0: self.ys[index + usize::from(!self.downward)],
            y1: self.ys[index + usize::from(self.downward)],
        }
    }

    fn at_y(&self, y: f64, height: f64) -> Option<LayoutBox> {
        // Match the original multiply-then-compare pixel test exactly. The
        // world-coordinate fallback calls this with height=1 instead.
        let starts = if self.downward {
            &self.ys[..self.ys.len() - 1]
        } else {
            &self.ys[1..]
        };
        let index = if self.downward {
            starts
                .partition_point(|&y0| y0 * height <= y)
                .checked_sub(1)?
        } else {
            starts.partition_point(|&y0| y0 * height > y)
        };
        let &y0 = starts.get(index)?;
        let y1 = self.ys[index + usize::from(self.downward)];
        (y0 * height <= y && y < y1 * height).then(|| self.get(index))
    }
}

#[derive(Debug)]
enum KnownEntry {
    Box(LayoutBox),
    Thin(ThinBranches),
}

#[derive(Debug, Default)]
pub struct KnownBranches {
    entries: Vec<KnownEntry>,
    spare_ys: Vec<Vec<f64>>,
}

impl KnownBranches {
    fn clear(&mut self) {
        self.spare_ys.clear();
        for entry in self.entries.drain(..) {
            if let KnownEntry::Thin(mut run) = entry {
                run.ys.clear();
                self.spare_ys.push(run.ys);
            }
        }
    }

    fn positions(&mut self, remaining: usize) -> Vec<f64> {
        self.spare_ys
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(remaining.min(4096) + 1))
    }
    fn push(&mut self, b: LayoutBox) {
        self.entries.push(KnownEntry::Box(b));
    }
    fn insert(&mut self, index: usize, b: LayoutBox) {
        self.entries.insert(index, KnownEntry::Box(b));
    }

    fn flush(&mut self, run: Option<ThinBranches>) {
        if let Some(run) = run {
            self.entries.push(KnownEntry::Thin(run));
        }
    }

    /// Materialization is a diagnostic operation, never part of a frame.
    pub fn materialize(&self) -> Vec<LayoutBox> {
        self.entries
            .iter()
            .flat_map(|entry| match entry {
                KnownEntry::Box(b) => vec![b.clone()],
                KnownEntry::Thin(run) => (0..run.ys.len() - 1).map(|i| run.get(i)).collect(),
            })
            .collect()
    }

    fn at_y(&self, y: f64, height: f64) -> impl Iterator<Item = LayoutBox> + '_ {
        self.entries.iter().filter_map(move |entry| match entry {
            KnownEntry::Box(b) => Some(b.clone()),
            KnownEntry::Thin(run) => run.at_y(y, height),
        })
    }
}

impl Frame {
    /// Release listing snapshots but keep the allocation for the next view.
    pub(crate) fn clear(&mut self) {
        self.boxes.clear();
        self.known.clear();
        self.walked.clear();
        self.unexpanded.clear();
        self.boundaries.clear();
        self.cover = None;
        self.origin_top = 0.0;
        self.origin_bottom = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Placement {
    pub ln_mass: f64,
    pub x1: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Extent {
    pub top: f64,
    pub bottom: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnchorPosition {
    pub anchor: Boundary,
    pub a: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PickResult {
    pub r#box: LayoutBox,
    pub exact: bool,
}

/// ln of the view's height in origin heights: the view spans e^-x of the
/// origin. Only the root bounds x, at 0.
fn log_view(cam: Camera) -> f64 {
    -cam.x
}

/// w * s, where a zero weight stays zero even at an infinite scale.
fn scaled(weight: f64, scale: f64) -> f64 {
    if weight == 0.0 {
        0.0
    } else {
        weight * scale
    }
}

/// The slot of `node` among its parent's slots under `lens`.
fn slot_in(tree: &Tree, lens: Lens, node: NodeId) -> Result<usize> {
    let parent = tree
        .parent(node)?
        .ok_or_else(|| "the root has no parent slot".to_string())?;
    let branch = tree
        .branch(node)?
        .ok_or_else(|| format!("non-root node {node} has no branch"))?;
    tree.slots(parent, lens)?
        .slot_of(branch)
        .ok_or_else(|| format!("node {node} is not listed under measure {}", lens.p))
}

/// Canonical spelling of a boundary: a node's bottom is its next sibling's
/// top, up to the origin, whose bottom is its last slot.
pub fn normalize(tree: &Tree, lens: Lens, origin: NodeId, boundary: Boundary) -> Result<Boundary> {
    let Boundary { mut node, mut slot } = boundary;
    while node != origin && slot >= tree.slots(node, lens)?.count() as u64 {
        slot = slot_in(tree, lens, node)? as u64 + 1;
        node = tree
            .parent(node)?
            .expect("a node below the origin has a parent");
    }
    if node == origin && slot >= tree.slots(node, lens)?.count() as u64 {
        slot = MAX_SAFE_SLOT;
    }
    Ok(Boundary { node, slot })
}

/// Move the anchor up to its deepest ancestor that p lists below the origin,
/// without mutation.
pub fn repair(tree: &Tree, lens: Lens, origin: NodeId, boundary: Boundary) -> Result<Boundary> {
    let chain = tree.chain(origin, boundary.node)?;
    for pair in chain.windows(2) {
        let parent = pair[0];
        let child = pair[1];
        let branch = tree
            .branch(child)?
            .ok_or_else(|| format!("non-root node {child} has no branch"))?;
        let listed = tree.slots(parent, lens)?.slot_of(branch).is_some();
        if !listed {
            return Ok(Boundary {
                node: parent,
                slot: 0,
            });
        }
    }
    Ok(boundary)
}

struct Walk {
    chain: Vec<NodeId>,
    slots: Vec<usize>,
    ln_masses: Vec<f64>,
}

/// The chain from the origin to the anchor, each node's slot in its parent,
/// and each node's ln p mass below the origin.
fn walk(tree: &Tree, lens: Lens, cam: Camera) -> Result<Walk> {
    let chain = tree.chain(cam.origin, cam.anchor.node)?;
    let mut slots = vec![0; chain.len()];
    let mut ln_masses = vec![0.0; chain.len()];
    for i in 1..chain.len() {
        slots[i] = slot_in(tree, lens, chain[i])?;
        let weights = tree.slots(chain[i - 1], lens)?;
        ln_masses[i] = ln_masses[i - 1] + weights.ln_p(slots[i]);
    }
    Ok(Walk {
        chain,
        slots,
        ln_masses,
    })
}

#[derive(Debug)]
struct Run {
    node: NodeId,
    first: usize,
    count: u64,
    p: f64,
    cross: f64,
    y0: f64,
    y1: f64,
    x: f64,
    ln_mass: f64,
}

struct Layout<'a> {
    tree: &'a Tree,
    lens: Lens,
    cam: Camera,
    right: f64,
    merge_units: f64,
    frame: Frame,
    cover_depth: usize,
}

impl Layout<'_> {
    fn consider_cover(&mut self, node: NodeId, y0: f64, y1: f64, x: f64) -> Result<()> {
        if y0 <= 1e-12 && y1 >= 1.0 - 1e-12 && x <= self.cam.x + 1e-12 {
            let depth = self.tree.depth(node)? + 1;
            if depth > self.cover_depth {
                self.cover_depth = depth;
                self.frame.cover = Some(Cover {
                    node,
                    y: y0.min(0.0),
                });
            }
        }
        Ok(())
    }

    fn boundary(&mut self, node: NodeId, slot: u64, y: f64) -> Result<()> {
        if (0.0..=1.0).contains(&y) {
            let boundary = normalize(
                self.tree,
                self.lens,
                self.cam.origin,
                Boundary { node, slot },
            )?;
            self.frame.boundaries.push(BoundaryPosition { boundary, y });
        }
        Ok(())
    }

    fn push(&mut self, layout_box: LayoutBox) {
        if layout_box.x1 > self.cam.x
            && layout_box.x0 < self.right
            && layout_box.y1 > 0.0
            && layout_box.y0 < 1.0
        {
            self.frame.boxes.push(layout_box);
        }
    }

    fn flush(&mut self, run: Option<Run>) {
        let Some(run) = run else { return };
        self.push(LayoutBox {
            node: run.node,
            slot: run.first as u64,
            count: run.count,
            kind: BoxKind::Merged,
            branch: None,
            p: run.p,
            // Effective q keeps the run's area equal to Σ p_i (-ln q_i).
            q: (-run.cross / run.p).exp(),
            ln_mass: run.ln_mass + run.p.ln(),
            x0: run.x,
            x1: run.x + run.cross / run.p,
            y0: run.y0,
            y1: run.y1,
        });
    }

    /// Lay out one visible slot and recurse from its edge nearest the origin.
    #[allow(clippy::too_many_arguments)]
    fn slot_box(
        &mut self,
        node: NodeId,
        slots: &Slots,
        slot: usize,
        y0: f64,
        y1: f64,
        scale: f64,
        x: f64,
        ln_mass: f64,
        downward: bool,
    ) -> Result<()> {
        let pw = slots.p(slot);
        let qw = slots.q(slot);
        let x1 = x + slots.nats_q(slot);
        let branch = slots.branch(slot);
        let child = match branch {
            Some(branch) => self.tree.child_id(node, branch)?,
            None => None,
        };
        let kind = match (branch, child) {
            (None, _) => BoxKind::Tail,
            (Some(_), Some(id)) if self.tree.expanded(id, self.lens.p)? => BoxKind::Node,
            _ => BoxKind::Leaf,
        };
        let child_mass = ln_mass + slots.ln_p(slot);
        if let Some(child) = child {
            self.consider_cover(child, y0, y1, x1)?;
        }
        let b = LayoutBox {
            node,
            slot: slot as u64,
            count: 1,
            kind,
            branch,
            p: pw,
            q: qw,
            ln_mass: child_mass,
            x0: x,
            x1,
            y0,
            y1,
        };
        self.frame.known.push(b.clone());
        self.push(b);
        self.boundary(node, slot as u64, y0)?;
        self.boundary(node, slot as u64 + 1, y1)?;
        if x1 >= self.right {
            return Ok(());
        }
        if kind == BoxKind::Leaf {
            self.frame.unexpanded.push(Unexpanded {
                node,
                branch: branch.expect("a leaf slot has a branch"),
                ln_mass: child_mass,
                x1,
            });
        }
        if kind != BoxKind::Node {
            return Ok(());
        }
        let child = child.expect("a node slot has a child");
        if downward {
            self.down(child, 0, y0, scaled(pw, scale), x1, child_mass)?;
        } else {
            let child_slots = self.tree.slots(child, self.lens)?;
            self.up(
                child,
                child_slots.count() as isize - 1,
                y1,
                scaled(pw, scale),
                x1,
                child_mass,
            )?;
        }
        Ok(())
    }

    /// Slots from `start` onward, whose top edge is at `y`.
    fn down(
        &mut self,
        node: NodeId,
        start: usize,
        mut y: f64,
        scale: f64,
        x: f64,
        ln_mass: f64,
    ) -> Result<()> {
        if x >= self.right {
            return Ok(());
        }
        self.frame.walked.push(Walked { node, ln_mass, x });
        let slots = self.tree.slots(node, self.lens)?;
        let mut thin: Option<ThinBranches> = None;
        let expanded = Arc::new(self.tree.expanded_branches(node, self.lens.p)?);
        let mut run: Option<Run> = None;
        let mut slot = start;
        while slot < slots.count() && y < 1.0 {
            let pw = slots.p(slot);
            let y1 = y + scaled(pw, scale);
            if y1 > 0.0 {
                if y1 - y < self.merge_units {
                    if thin.is_none() {
                        thin = Some(ThinBranches {
                            node,
                            slots: Arc::clone(&slots),
                            expanded: Arc::clone(&expanded),
                            first: slot,
                            downward: true,
                            ys: {
                                let mut ys = self.frame.known.positions(slots.count() - slot);
                                ys.push(y);
                                ys
                            },
                            x,
                            ln_mass,
                        });
                    }
                    thin.as_mut().expect("thin run initialized").ys.push(y1);
                    let current = run.get_or_insert(Run {
                        node,
                        first: slot,
                        count: 0,
                        p: 0.0,
                        cross: 0.0,
                        y0: y,
                        y1: y,
                        x,
                        ln_mass,
                    });
                    current.count += 1;
                    current.p += pw;
                    if pw > 0.0 {
                        current.cross += pw * slots.nats_q(slot);
                    }
                    current.y1 = y1;
                    if current.y1 - current.y0 >= self.merge_units {
                        self.flush(run.take());
                    }
                } else {
                    self.flush(run.take());
                    self.frame.known.flush(thin.take());
                    self.slot_box(node, &slots, slot, y, y1, scale, x, ln_mass, true)?;
                }
            }
            y = y1;
            slot += 1;
        }
        self.flush(run.take());
        self.frame.known.flush(thin.take());
        Ok(())
    }

    /// Slots from `start` back to the first, whose bottom edge is at `y`.
    fn up(
        &mut self,
        node: NodeId,
        start: isize,
        mut y: f64,
        scale: f64,
        x: f64,
        ln_mass: f64,
    ) -> Result<()> {
        if x >= self.right {
            return Ok(());
        }
        self.frame.walked.push(Walked { node, ln_mass, x });
        let slots = self.tree.slots(node, self.lens)?;
        let mut thin: Option<ThinBranches> = None;
        let expanded = Arc::new(self.tree.expanded_branches(node, self.lens.p)?);
        let mut run: Option<Run> = None;
        let mut slot = start;
        while slot >= 0 && y > 0.0 {
            let index = slot as usize;
            let pw = slots.p(index);
            let y0 = y - scaled(pw, scale);
            if y0 < 1.0 {
                if y - y0 < self.merge_units {
                    if thin.is_none() {
                        thin = Some(ThinBranches {
                            node,
                            slots: Arc::clone(&slots),
                            expanded: Arc::clone(&expanded),
                            first: index,
                            downward: false,
                            ys: {
                                let mut ys = self.frame.known.positions(index + 1);
                                ys.push(y);
                                ys
                            },
                            x,
                            ln_mass,
                        });
                    }
                    thin.as_mut().expect("thin run initialized").ys.push(y0);
                    let current = run.get_or_insert(Run {
                        node,
                        first: index,
                        count: 0,
                        p: 0.0,
                        cross: 0.0,
                        y0: y,
                        y1: y,
                        x,
                        ln_mass,
                    });
                    current.first = index;
                    current.count += 1;
                    current.p += pw;
                    if pw > 0.0 {
                        current.cross += pw * slots.nats_q(index);
                    }
                    current.y0 = y0;
                    if current.y1 - current.y0 >= self.merge_units {
                        self.flush(run.take());
                    }
                } else {
                    self.flush(run.take());
                    self.frame.known.flush(thin.take());
                    self.slot_box(node, &slots, index, y0, y, scale, x, ln_mass, false)?;
                }
            }
            y = y0;
            slot -= 1;
        }
        self.flush(run.take());
        self.frame.known.flush(thin.take());
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
    frame: Frame,
) -> Result<Frame> {
    let lambda = log_view(cam);
    let mut context = Layout {
        tree,
        lens,
        cam,
        right: cam.x + view.width / cam.k,
        merge_units: merge_px / view.height,
        frame,
        cover_depth: 0,
    };

    let Walk {
        chain,
        slots: chain_slots,
        ln_masses,
    } = walk(tree, lens, cam)?;
    let depth = chain.len() - 1;
    let mut xs = vec![0.0; chain.len()];
    for i in 1..=depth {
        let slots = tree.slots(chain[i - 1], lens)?;
        xs[i] = xs[i - 1] + slots.nats_q(chain_slots[i]);
    }

    let anchor = chain[depth];
    let anchor_slots = tree.slots(anchor, lens)?;
    let slot = cam.anchor.slot.min(anchor_slots.count() as u64) as usize;
    let mut scale = (ln_masses[depth] - lambda).exp();
    let mut top = cam.a - scaled(anchor_slots.prefix(slot), scale);
    let mut bottom = cam.a
        + scaled(
            if anchor_slots.count() == 0 {
                1.0
            } else {
                anchor_slots.suffix(slot)
            },
            scale,
        );
    context.boundary(anchor, slot as u64, cam.a)?;
    context.down(anchor, slot, cam.a, scale, xs[depth], ln_masses[depth])?;
    context.up(
        anchor,
        slot as isize - 1,
        cam.a,
        scale,
        xs[depth],
        ln_masses[depth],
    )?;

    // Walking from the anchor upward makes the first qualifying node the deepest.
    context.consider_cover(anchor, top, bottom, xs[depth])?;

    for i in (0..depth).rev() {
        let owner = chain[i];
        let slots = tree.slots(owner, lens)?;
        let slot = chain_slots[i + 1];
        let x = xs[i];
        scale = (ln_masses[i] - lambda).exp();
        // Ancestor boxes stay available even when their own rectangle is
        // offscreen: they are the known branches used by null-space picking.
        let ancestor = LayoutBox {
            node: owner,
            slot: slot as u64,
            count: 1,
            kind: BoxKind::Node,
            branch: tree.branch(chain[i + 1])?,
            p: slots.p(slot),
            q: slots.q(slot),
            ln_mass: ln_masses[i + 1],
            x0: x,
            x1: xs[i + 1],
            y0: top,
            y1: bottom,
        };
        context.frame.known.push(ancestor.clone());
        context.frame.boxes.push(ancestor);
        context.boundary(owner, slot as u64, top)?;
        context.boundary(owner, slot as u64 + 1, bottom)?;
        context.down(owner, slot + 1, bottom, scale, x, ln_masses[i])?;
        context.up(owner, slot as isize - 1, top, scale, x, ln_masses[i])?;
        top -= scaled(slots.prefix(slot), scale);
        bottom += scaled(slots.suffix(slot + 1), scale);

        context.consider_cover(owner, top, bottom, x)?;
        // Once this node covers the view and its own box lies left of it, every
        // remaining ancestor box and sibling is off screen.
        if top <= 0.0 && bottom >= 1.0 && x <= cam.x {
            break;
        }
    }

    context.frame.origin_top = top;
    context.frame.origin_bottom = bottom;
    // The origin remains selectable before any expansion has arrived.
    context.frame.known.insert(
        0,
        LayoutBox {
            node: cam.origin,
            slot: 0,
            count: 0,
            kind: BoxKind::Node,
            branch: None,
            p: 1.0,
            q: 1.0,
            ln_mass: 0.0,
            x0: 0.0,
            x1: 0.0,
            y0: top.min(0.0),
            y1: bottom.max(1.0),
        },
    );
    Ok(context.frame)
}

/// `id`'s ln p mass and right edge in nats below `origin`, -Σ ln q along the way.
pub fn place(tree: &Tree, lens: Lens, origin: NodeId, id: NodeId) -> Result<Placement> {
    let mut placement = Placement {
        ln_mass: 0.0,
        x1: 0.0,
    };
    for node in tree.chain(origin, id)?.into_iter().skip(1) {
        let parent = tree
            .parent(node)?
            .expect("a node below the origin has a parent");
        let slots = tree.slots(parent, lens)?;
        let slot = slot_in(tree, lens, node)?;
        placement.ln_mass += slots.ln_p(slot);
        placement.x1 += slots.nats_q(slot);
    }
    Ok(placement)
}

/// The origin's interval edges in view heights, walking the whole anchor chain.
pub fn extent(tree: &Tree, lens: Lens, cam: Camera) -> Result<Extent> {
    let lambda = log_view(cam);
    let Walk {
        chain,
        slots: chain_slots,
        ln_masses,
    } = walk(tree, lens, cam)?;
    let depth = chain.len() - 1;
    let anchor = tree.slots(chain[depth], lens)?;
    let slot = cam.anchor.slot.min(anchor.count() as u64) as usize;
    let mut scale = (ln_masses[depth] - lambda).exp();
    let mut top = cam.a - scaled(anchor.prefix(slot), scale);
    let mut bottom = cam.a
        + scaled(
            if anchor.count() == 0 {
                1.0
            } else {
                anchor.suffix(slot)
            },
            scale,
        );
    for i in (0..depth).rev() {
        let owner = tree.slots(chain[i], lens)?;
        let slot = chain_slots[i + 1];
        scale = (ln_masses[i] - lambda).exp();
        top -= scaled(owner.prefix(slot), scale);
        bottom += scaled(owner.suffix(slot + 1), scale);
    }
    Ok(Extent { top, bottom })
}

/// Re-anchor at the in-view boundary nearest the view's center.
pub fn reanchor(frame: &Frame) -> Option<AnchorPosition> {
    let mut best: Option<&BoundaryPosition> = None;
    for candidate in &frame.boundaries {
        if best.is_none_or(|current| (candidate.y - 0.5).abs() < (current.y - 0.5).abs()) {
            best = Some(candidate);
        }
    }
    best.map(|best| AnchorPosition {
        anchor: best.boundary,
        a: best.y,
    })
}

/// Keep the view inside the root's interval, returning only the new anchor offset.
pub fn clamp_position(tree: &Tree, lens: Lens, cam: Camera) -> Result<f64> {
    // Only the root's interval is all there is; beyond any other origin lies
    // its parent, unknown until fetched.
    if cam.origin != tree.root() {
        return Ok(cam.a);
    }
    let Extent { top, bottom } = extent(tree, lens, cam)?;
    if top > 0.0 {
        Ok(cam.a - top)
    } else if bottom < 1.0 {
        Ok(cam.a + 1.0 - bottom)
    } else {
        Ok(cam.a)
    }
}

const OVERSCAN: f64 = 64.0;

struct ScreenRect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

fn screen_rect(layout_box: &LayoutBox, x: f64, k: f64, view: Viewport) -> ScreenRect {
    let clamp = |value: f64, extent: f64| value.max(-OVERSCAN).min(extent + OVERSCAN);
    ScreenRect {
        x0: clamp((layout_box.x0 - x) * k, view.width),
        x1: clamp((layout_box.x1 - x) * k, view.width),
        y0: clamp(layout_box.y0 * view.height, view.height),
        y1: clamp(layout_box.y1 * view.height, view.height),
    }
}

fn vertical_distance(layout_box: &LayoutBox, y: f64) -> f64 {
    if y < layout_box.y0 {
        layout_box.y0 - y
    } else if y >= layout_box.y1 {
        y - layout_box.y1
    } else {
        0.0
    }
}

fn horizontal_distance(layout_box: &LayoutBox, x: f64) -> f64 {
    if x < layout_box.x0 {
        layout_box.x0 - x
    } else if x > layout_box.x1 {
        x - layout_box.x1
    } else {
        0.0
    }
}

/// The box under a screen point, for a frame drawn with left edge `left` nats
/// and scale `k` pixels per nat.
pub fn pick(
    frame: &Frame,
    left: f64,
    k: f64,
    view: Viewport,
    x_px: f64,
    y_px: f64,
) -> Option<PickResult> {
    if x_px < 0.0 || x_px >= view.width || y_px < 0.0 || y_px >= view.height {
        return None;
    }

    // Exact hits retain frame order and the renderer's overscan semantics.
    for layout_box in &frame.boxes {
        if layout_box.kind == BoxKind::Merged {
            continue;
        }
        let rect = screen_rect(layout_box, left, k, view);
        if x_px >= rect.x0 && x_px < rect.x1 && y_px >= rect.y0 && y_px < rect.y1 {
            return Some(PickResult {
                r#box: layout_box.clone(),
                exact: layout_box.branch.is_some(),
            });
        }
    }

    for layout_box in frame.known.at_y(y_px, view.height) {
        let rect = screen_rect(&layout_box, left, k, view);
        if x_px >= rect.x0 && x_px < rect.x1 && y_px >= rect.y0 && y_px < rect.y1 {
            return Some(PickResult {
                r#box: layout_box.clone(),
                exact: layout_box.branch.is_some(),
            });
        }
    }

    let x = left + x_px / k;
    let y = y_px / view.height;
    let candidates = || {
        frame
            .boxes
            .iter()
            .filter(|b| b.kind != BoxKind::Merged)
            .cloned()
            .chain(frame.known.at_y(y, 1.0))
    };
    let mut left: Option<LayoutBox> = None;
    let mut right: Option<LayoutBox> = None;
    for b in candidates() {
        if !(b.y0 <= y && y < b.y1) {
            continue;
        }
        if b.x0 <= x {
            if left.as_ref().is_none_or(|best| b.x0 >= best.x0) {
                left = Some(b);
            }
        } else if right.as_ref().is_none_or(|best| b.x0 < best.x0) {
            right = Some(b);
        }
    }
    if let Some(b) = left.or(right) {
        return Some(PickResult {
            r#box: b,
            exact: false,
        });
    }

    // Best effort: vertical distance first, horizontal distance only breaks ties.
    let mut best: Option<(LayoutBox, f64, f64)> = None;
    for b in candidates() {
        let vertical = vertical_distance(&b, y);
        let horizontal = horizontal_distance(&b, x);
        if best
            .as_ref()
            .is_none_or(|(_, bv, bh)| vertical < *bv || (vertical == *bv && horizontal < *bh))
        {
            best = Some((b, vertical, horizontal));
        }
    }
    best.map(|(b, _, _)| PickResult {
        r#box: b,
        exact: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{child_hash, Listing};

    const VIEW: Viewport = Viewport {
        width: 1000.0,
        height: 800.0,
    };
    const LENS: Lens = Lens { p: 0, q: 0 };

    fn listing(weights: &[f64]) -> Listing {
        Listing {
            horizon: 40.0,
            branches: (0..weights.len() as u32).collect(),
            ln_weights: weights.iter().map(|weight| weight.ln()).collect(),
            leftover: 1.0 - weights.iter().sum::<f64>(),
            complete: 40.0,
        }
    }

    fn chain_tree(weights: &[f64], path: &[u32]) -> Tree {
        let mut tree = Tree::new(1);
        let children = listing(weights);
        tree.set_listing(0, 0, children.clone()).unwrap();
        let mut node = 0;
        for &branch in path {
            node = tree.child(node, branch).unwrap();
            tree.set_listing(node, 0, children.clone()).unwrap();
        }
        tree
    }

    fn near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-11,
            "{actual} differs from {expected}"
        );
    }

    #[test]
    fn layout_partitions_by_p_sizes_by_q_and_merges_thin_runs() {
        let tree = chain_tree(&[0.5, 0.25, 0.25], &[]);
        let cam = Camera {
            origin: 0,
            x: 0.0,
            k: 100.0,
            anchor: Boundary { node: 0, slot: 0 },
            a: 0.0,
        };
        let frame = layout(&tree, LENS, cam, VIEW, 1.0).unwrap();
        let spans = frame
            .boxes
            .iter()
            .map(|b| (b.slot, b.y0, b.y1, b.x1 - b.x0))
            .collect::<Vec<_>>();
        assert_eq!(spans.len(), 3);
        for (actual, expected) in spans.iter().zip([
            (0, 0.0, 0.5, std::f64::consts::LN_2),
            (1, 0.5, 0.75, 4.0f64.ln()),
            (2, 0.75, 1.0, 4.0f64.ln()),
        ]) {
            assert_eq!(actual.0, expected.0);
            near(actual.1, expected.1);
            near(actual.2, expected.2);
            near(actual.3, expected.3);
        }
        near(frame.origin_top, 0.0);
        near(frame.origin_bottom, 1.0);
        assert_eq!(frame.cover.map(|cover| cover.node), Some(0));

        let mut thin_listing = listing(&[0.5, 0.488, 0.004, 0.004, 0.004]);
        thin_listing.leftover = 0.0;
        let mut thin_tree = Tree::new(1);
        thin_tree.set_listing(0, 0, thin_listing).unwrap();
        let merged = layout(&thin_tree, LENS, cam, VIEW, 8.0).unwrap();
        let run = merged
            .boxes
            .iter()
            .find(|b| b.kind == BoxKind::Merged)
            .expect("thin slots merge");
        assert_eq!((run.slot, run.count), (2, 3));
        near(run.p, 0.012);
        near(run.y0, 0.988);
        near(run.y1, 1.0);
        assert_eq!(run.branch, None);
    }

    #[test]
    fn deep_layout_retains_anchor_precision_and_finds_deepest_cover() {
        let path = (0..80).map(|i| i % 2).collect::<Vec<_>>();
        let tree = chain_tree(&[0.5, 0.5], &path);
        let id = |path: &[u32]| {
            path.iter()
                .fold(0, |node, &branch| child_hash(node, branch))
        };
        let (deepest, above) = (id(&path), id(&path[..79]));
        let cam = Camera {
            origin: 0,
            x: 80.0 * std::f64::consts::LN_2,
            k: 100.0,
            anchor: Boundary {
                node: deepest,
                slot: 1,
            },
            a: 0.5,
        };
        let frame = layout(&tree, LENS, cam, VIEW, 1.0).unwrap();
        let mut children = frame
            .boxes
            .iter()
            .filter(|b| b.node == deepest)
            .collect::<Vec<_>>();
        children.sort_by_key(|b| b.slot);
        assert_eq!(children.len(), 2);
        near(children[0].y0, 0.0);
        near(children[0].y1, 0.5);
        near(children[1].y0, 0.5);
        near(children[1].y1, 1.0);
        let own = frame
            .boxes
            .iter()
            .find(|b| b.node == above && b.slot == u64::from(path[79]))
            .unwrap();
        near(own.y0, 0.0);
        near(own.y1, 1.0);
        let cover = frame.cover.expect("an ancestor covers the view");
        assert!(cover.y <= 0.0);

        let shallow_tree = chain_tree(&[0.5, 0.5], &[0]);
        let shallow_cam = Camera {
            origin: 0,
            x: std::f64::consts::LN_2,
            anchor: Boundary {
                node: id(&[0]),
                slot: 0,
            },
            a: 0.0,
            ..cam
        };
        let shallow = layout(&shallow_tree, LENS, shallow_cam, VIEW, 1.0).unwrap();
        assert_eq!(
            shallow.cover,
            Some(Cover {
                node: id(&[0]),
                y: 0.0
            })
        );
        let root = layout(
            &shallow_tree,
            LENS,
            Camera {
                origin: 0,
                x: 0.0,
                anchor: Boundary { node: 0, slot: 0 },
                a: 0.0,
                ..cam
            },
            VIEW,
            1.0,
        )
        .unwrap();
        assert_eq!(root.cover, Some(Cover { node: 0, y: 0.0 }));
    }

    #[test]
    fn boundary_helpers_repair_clamp_place_and_reanchor() {
        let mut tree = Tree::new(2);
        let children = listing(&[0.5, 0.5]);
        tree.set_listing(0, 0, children.clone()).unwrap();
        let first = tree.child(0, 1).unwrap();
        tree.set_listing(first, 0, children.clone()).unwrap();
        let second = tree.child(first, 0).unwrap();
        tree.set_listing(second, 0, children.clone()).unwrap();
        for &node in &[0, first, second] {
            tree.set_listing(node, 1, listing(&[1.0])).unwrap();
        }
        assert_eq!(
            repair(
                &tree,
                Lens { p: 1, q: 1 },
                0,
                Boundary {
                    node: second,
                    slot: 1
                }
            )
            .unwrap(),
            Boundary { node: 0, slot: 0 }
        );
        assert_eq!(
            normalize(
                &tree,
                LENS,
                0,
                Boundary {
                    node: first,
                    slot: 2
                }
            )
            .unwrap(),
            Boundary {
                node: 0,
                slot: MAX_SAFE_SLOT,
            }
        );
        let placed = place(&tree, LENS, 0, second).unwrap();
        near(placed.ln_mass, -2.0 * std::f64::consts::LN_2);
        near(placed.x1, 2.0 * std::f64::consts::LN_2);

        let clamp_tree = chain_tree(&[0.5, 0.5], &[0]);
        let cam = Camera {
            origin: 0,
            x: 0.5,
            k: 100.0,
            anchor: Boundary {
                node: child_hash(0, 0),
                slot: 0,
            },
            a: 0.4,
        };
        let a = clamp_position(&clamp_tree, LENS, cam).unwrap();
        near(a, 0.0);
        near(
            extent(&clamp_tree, LENS, Camera { a, ..cam }).unwrap().top,
            0.0,
        );

        let frame = Frame {
            boundaries: vec![
                BoundaryPosition {
                    boundary: Boundary { node: 3, slot: 1 },
                    y: 0.2,
                },
                BoundaryPosition {
                    boundary: Boundary { node: 4, slot: 2 },
                    y: 0.6,
                },
            ],
            ..Frame::default()
        };
        assert_eq!(
            reanchor(&frame),
            Some(AnchorPosition {
                anchor: Boundary { node: 4, slot: 2 },
                a: 0.6,
            })
        );
    }

    #[test]
    fn an_origin_stands_as_the_root_without_any_ancestor_listing() {
        let mut tree = Tree::new(1);
        let origin = tree.descend(0, &[7; 3000]).unwrap();
        tree.set_listing(origin, 0, listing(&[0.5, 0.5])).unwrap();
        let first = tree.child(origin, 0).unwrap();
        tree.set_listing(first, 0, listing(&[0.5, 0.5])).unwrap();
        let cam = Camera {
            origin,
            x: 0.0,
            k: 100.0,
            anchor: Boundary {
                node: first,
                slot: 0,
            },
            a: 0.3,
        };
        // Nothing bounds a view below the root: what lies past the origin is
        // its parent, unknown until fetched, and stays empty.
        assert_eq!(clamp_position(&tree, LENS, cam).unwrap(), 0.3);
        let outside = layout(&tree, LENS, cam, VIEW, 1.0).unwrap();
        near(outside.origin_top, 0.3);
        near(outside.origin_bottom, 1.3);
        assert!(outside.cover.is_none());
        let cam = Camera { a: 0.0, ..cam };
        let frame = layout(&tree, LENS, cam, VIEW, 1.0).unwrap();
        near(frame.origin_top, 0.0);
        near(frame.origin_bottom, 1.0);
        assert_eq!(frame.cover.unwrap().node, origin);
        assert_eq!(frame.known.materialize()[0].node, origin);
        let placed = place(&tree, LENS, origin, first).unwrap();
        near(placed.ln_mass, -std::f64::consts::LN_2);
        assert_eq!(
            normalize(
                &tree,
                LENS,
                origin,
                Boundary {
                    node: first,
                    slot: 2
                }
            )
            .unwrap(),
            Boundary {
                node: origin,
                slot: 1
            }
        );
        assert_eq!(
            repair(
                &tree,
                LENS,
                origin,
                Boundary {
                    node: first,
                    slot: 1
                }
            )
            .unwrap(),
            Boundary {
                node: first,
                slot: 1
            }
        );
    }

    #[test]
    fn offscreen_leaf_still_requests_its_unexpanded_children() {
        let tree = chain_tree(&[0.5, 0.5], &[]);
        let cam = Camera {
            origin: 0,
            x: std::f64::consts::LN_2,
            k: 100.0,
            anchor: Boundary { node: 0, slot: 0 },
            a: 0.0,
        };
        let frame = layout(&tree, LENS, cam, VIEW, 1.0).unwrap();
        assert!(!frame.boxes.iter().any(|b| b.node == 0 && b.slot == 0));
        assert!(frame.unexpanded.iter().any(|child| {
            child.node == 0
                && child.branch == 0
                && (child.ln_mass + std::f64::consts::LN_2).abs() < 1e-12
                && (child.x1 - std::f64::consts::LN_2).abs() < 1e-12
        }));
    }

    fn picking_box(node: NodeId, x0: f64, x1: f64, y0: f64, y1: f64) -> LayoutBox {
        LayoutBox {
            node,
            slot: 0,
            count: 1,
            kind: BoxKind::Leaf,
            branch: Some(node as u32),
            p: 1.0,
            q: 1.0,
            ln_mass: 0.0,
            x0,
            x1,
            y0,
            y1,
        }
    }

    fn picking_frame(boxes: Vec<LayoutBox>) -> Frame {
        Frame {
            boxes,
            ..Frame::default()
        }
    }

    const PICK_VIEW: Viewport = Viewport {
        width: 100.0,
        height: 100.0,
    };

    #[test]
    fn picking_uses_first_exact_box_in_frame_order() {
        let frame = picking_frame(vec![
            picking_box(1, 0.0, 1.0, 0.0, 1.0),
            picking_box(2, 0.0, 1.0, 0.0, 1.0),
        ]);
        let hit = pick(&frame, 0.0, 100.0, PICK_VIEW, 20.0, 20.0).unwrap();
        assert!(hit.exact);
        assert_eq!(hit.r#box.node, 1);
        assert!(pick(&frame, 0.0, 100.0, PICK_VIEW, 20.0, 100.0).is_none());
    }

    #[test]
    fn picking_fallback_considers_y_before_horizontal_proximity() {
        let frame = picking_frame(vec![
            // Horizontally containing, but in another vertical interval.
            picking_box(1, 0.29, 0.31, 0.0, 0.25),
            picking_box(2, 0.0, 0.1, 0.4, 0.6),
            // Deepest known branch at the cursor's y: greatest x0 <= 0.3.
            picking_box(3, 0.2, 0.25, 0.4, 0.6),
        ]);
        let hit = pick(&frame, 0.0, 100.0, PICK_VIEW, 30.0, 50.0).unwrap();
        assert!(!hit.exact);
        assert_eq!(hit.r#box.node, 3);

        let all_right = picking_frame(vec![
            picking_box(4, 0.4, 0.45, 0.4, 0.6),
            picking_box(5, 0.6, 0.65, 0.4, 0.6),
        ]);
        let hit = pick(&all_right, 0.0, 100.0, PICK_VIEW, 30.0, 50.0).unwrap();
        assert!(!hit.exact);
        assert_eq!(hit.r#box.node, 4);
    }

    #[test]
    fn picking_best_effort_is_vertical_then_horizontal() {
        let frame = picking_frame(vec![
            picking_box(1, 0.0, 0.1, 0.2, 0.3),
            picking_box(2, 0.29, 0.31, 0.7, 0.8),
        ]);
        let hit = pick(&frame, 0.0, 100.0, PICK_VIEW, 30.0, 50.0).unwrap();
        assert!(!hit.exact);
        assert_eq!(hit.r#box.node, 2);
    }
}
