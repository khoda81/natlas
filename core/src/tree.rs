//! The integer-sequence tree every client and server names alike: nodes by
//! hashed ids, and what each measure says of a node's children.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::distribution::{ln_between, Distribution, Item, Rank, Run, MAX_DESCENT};
use crate::point::Point;
use crate::Result;

/// A node's id, hashed from its branches by [`child_hash`] and shared by every
/// client and server.
pub type NodeId = u64;
pub const ROOT: NodeId = 0;

/// Serde for a node id: JavaScript receives a BigInt, since a 64-bit hash is
/// no exact Number.
pub mod node_id {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(id: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u128(u128::from(*id))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        u64::deserialize(deserializer)
    }
}

/// One of a node's 2^32 branches. Every branch exists; one a measure never
/// lists has weight zero. What a branch denotes is up to the reader.
pub type Branch = u32;

/// The measure whose order and weights split heights, and the one whose
/// surprisal sets widths.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lens {
    pub p: usize,
    pub q: usize,
}

/// A node's id shared by every client and server: hashed from its path, so
/// naming a node never needs the path itself. SplitMix64's finalizer is a
/// bijection, so siblings never collide; other collisions are left to chance
/// at 64 bits and refused when they happen.
pub fn child_hash(parent: u64, branch: Branch) -> u64 {
    let mut z = parent.wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(u64::from(branch) + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Debug)]
struct Node {
    /// Parent and the branch to this node; `None` only for the root.
    edge: Option<(NodeId, Branch)>,
    distributions: Vec<Option<Distribution>>,
}

/// A child as a lens sees it within its parent.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub rank: u32,
    /// p mass between the parent's top and the child's.
    pub before: f64,
    pub ln_p: f64,
    /// −ln q; infinite while q is unknown.
    pub nats_q: f64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Statistics {
    /// Over the known children.
    pub entropy: f64,
    pub cross: f64,
    /// Whether children or their q are still unknown, so both are lower bounds.
    pub bounded: bool,
}

#[derive(Debug)]
pub struct Tree {
    nodes: HashMap<NodeId, Node>,
    measures: usize,
}

impl Tree {
    pub fn new(measures: usize) -> Self {
        let mut tree = Self {
            nodes: HashMap::new(),
            measures,
        };
        tree.insert_node(ROOT, None);
        tree
    }

    fn insert_node(&mut self, id: NodeId, edge: Option<(NodeId, Branch)>) {
        let distributions = (0..self.measures).map(|_| None).collect();
        self.nodes.insert(
            id,
            Node {
                edge,
                distributions,
            },
        );
    }

    fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes
            .get(&id)
            .ok_or_else(|| format!("unknown node {id:#018x}"))
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node> {
        self.nodes
            .get_mut(&id)
            .ok_or_else(|| format!("unknown node {id:#018x}"))
    }

    /// The child of `parent` along `branch`, learned on first use.
    pub fn child(&mut self, parent: NodeId, branch: Branch) -> Result<NodeId> {
        self.node(parent)?;
        let id = child_hash(parent, branch);
        match self.nodes.get(&id) {
            Some(node) if node.edge != Some((parent, branch)) => {
                Err(format!("node id {id:#018x} collides"))
            }
            Some(_) => Ok(id),
            None => {
                self.insert_node(id, Some((parent, branch)));
                Ok(id)
            }
        }
    }

    /// Follow `path` down from `node`, learning only missing nodes.
    pub fn descend(&mut self, mut node: NodeId, path: &[Branch]) -> Result<NodeId> {
        self.node(node)?;
        for &branch in path {
            node = self.child(node, branch)?;
        }
        Ok(node)
    }

    /// Whether this tree has learned the node.
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn parent(&self, id: NodeId) -> Result<Option<NodeId>> {
        Ok(self.node(id)?.edge.map(|(parent, _)| parent))
    }

    pub fn branch(&self, id: NodeId) -> Result<Option<Branch>> {
        Ok(self.node(id)?.edge.map(|(_, branch)| branch))
    }

    /// Nodes from `origin` down to `id`, inclusive.
    pub fn chain(&self, origin: NodeId, id: NodeId) -> Result<Vec<NodeId>> {
        let mut chain = vec![id];
        let mut current = id;
        while current != origin {
            current = self.parent(current)?.ok_or_else(|| {
                format!("node {origin:#018x} is not an ancestor of node {id:#018x}")
            })?;
            chain.push(current);
        }
        chain.reverse();
        Ok(chain)
    }

    /// Branches from the root down to `id`.
    pub fn path(&self, id: NodeId) -> Result<Vec<Branch>> {
        let mut path = Vec::new();
        let mut current = self.node(id)?;
        while let Some((parent, branch)) = current.edge {
            path.push(branch);
            current = self.node(parent)?;
        }
        path.reverse();
        Ok(path)
    }

    fn check_measure(&self, measure: usize) -> Result<()> {
        if measure < self.measures {
            Ok(())
        } else {
            Err(format!("unknown measure {measure}"))
        }
    }

    /// What `measure` has said of the node's children; nothing for a node
    /// this tree has not learned.
    pub fn distribution(&self, id: NodeId, measure: usize) -> Result<Option<&Distribution>> {
        self.check_measure(measure)?;
        Ok(self
            .nodes
            .get(&id)
            .and_then(|node| node.distributions[measure].as_ref()))
    }

    /// Add a run a source sent, and whether anything changed.
    /// Add one answer's runs for the node, and whether anything changed.
    pub fn insert(&mut self, id: NodeId, measure: usize, runs: Vec<Run>) -> Result<bool> {
        self.check_measure(measure)?;
        let slot = &mut self.node_mut(id)?.distributions[measure];
        match slot {
            Some(distribution) => distribution.insert(runs),
            None if runs.is_empty() => Ok(false),
            None => {
                *slot = Some(Distribution::from_answer(runs)?);
                Ok(true)
            }
        }
    }

    /// Replace everything known of the node's children under `measure`.
    pub fn set(&mut self, id: NodeId, measure: usize, distribution: Distribution) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = Some(distribution);
        Ok(())
    }

    pub fn forget(&mut self, id: NodeId, measure: usize) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = None;
        Ok(())
    }

    /// −ln q of a child of `id`, infinite while unknown: an unknown width
    /// reaches past any view.
    pub fn nats_q(&self, id: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64> {
        if lens.p == lens.q {
            return Ok(-ln_p);
        }
        Ok(self
            .distribution(id, lens.q)?
            .and_then(|q| q.ln_p(branch))
            .map_or(f64::INFINITY, |ln_q| -ln_q))
    }

    /// Where p places the child along `branch` in `id`, if it does.
    pub fn entry(&self, id: NodeId, lens: Lens, branch: Branch) -> Result<Option<Entry>> {
        let Some(p) = self.distribution(id, lens.p)? else {
            return Ok(None);
        };
        let Some(rank) = p.rank(branch) else {
            return Ok(None);
        };
        let top = p.edge(rank).expect("a ranked branch has a known edge");
        let ln_p = p.ln_p(branch).expect("a ranked branch has a weight");
        Ok(Some(Entry {
            rank,
            before: ln_between(0.0, top).exp(),
            ln_p,
            nats_q: self.nats_q(id, lens, branch, ln_p)?,
        }))
    }

    pub fn statistics(&self, id: NodeId, lens: Lens) -> Result<Statistics> {
        let Some(p) = self.distribution(id, lens.p)? else {
            return Ok(Statistics {
                entropy: 0.0,
                cross: 0.0,
                bounded: true,
            });
        };
        let mut entropy = 0.0;
        let mut cross = 0.0;
        let mut bounded = !p.complete();
        for run in p.runs() {
            entropy += run.ln_surprisal(0, run.len()).exp();
            if lens.p != lens.q {
                for at in 0..run.len() {
                    let ln_p = run.ln_p(at);
                    let nats = self.nats_q(id, lens, run.branch(at), ln_p)?;
                    if nats.is_finite() {
                        cross += ln_p.exp() * nats;
                    } else {
                        bounded = true;
                    }
                }
            }
        }
        if lens.p == lens.q {
            cross = entropy;
        }
        Ok(Statistics {
            entropy,
            cross,
            bounded,
        })
    }
}

/// How an Expand reads the tree below its node: down a path, placing each of
/// its branches, and then within the node it ends at. Without a `within`,
/// the path's last branch names the node it reaches, which is not read.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Read {
    #[serde(default)]
    pub path: Vec<Branch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<Within>,
}

impl Read {
    /// Down these branches, placing each.
    pub fn path(path: Vec<Branch>) -> Self {
        Self { path, within: None }
    }

    /// The children of the node holding the points sampled from these parts.
    pub fn points(parts: Vec<Part>) -> Self {
        Within::Points(parts).into()
    }

    /// These children of the node.
    pub fn branches(branches: Vec<Branch>) -> Self {
        Within::Branches(branches).into()
    }
}

/// A read of the node itself.
impl From<Within> for Read {
    fn from(within: Within) -> Self {
        Self {
            path: Vec::new(),
            within: Some(within),
        }
    }
}

/// What an Expand reads of the node its path ends at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Within {
    /// The children holding the points sampled from these parts, and down
    /// through a child holding all of them.
    Points(Vec<Part>),
    /// These children of the node.
    Branches(Vec<Branch>),
}

/// `count` points of a node, evenly through the part of it from `top` down
/// e^`ln_mass` of its mass, each in the middle of its share. The top is
/// exact at any depth; a part of no mass is its top alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// The fraction of the node's mass above the part, in base 256.
    #[serde(with = "bytes")]
    pub top: Vec<u8>,
    pub ln_mass: f64,
    pub count: u32,
}

impl Part {
    /// Its points that are the node's: a part may reach past its bottom.
    fn points(&self) -> impl Iterator<Item = Point> + '_ {
        let top = Point::new(&self.top);
        (0..self.count).filter_map(move |i| {
            let share = (f64::from(i) + 0.5) / f64::from(self.count);
            top.moved(true, share, self.ln_mass)
        })
    }
}

/// Digits cross to JavaScript as a Uint8Array.
pub(crate) mod bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(digits: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(digits)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        Vec::deserialize(deserializer)
    }
}

/// One request of the protocol's Expand walked through a tree, one node at a
/// time. A recording answers from what it has; a server computes each
/// missing distribution as the walk asks for it.
#[derive(Debug)]
pub struct Walk {
    node: NodeId,
    path: Vec<Branch>,
    /// Branches of the path placed so far.
    at: usize,
    within: Option<Reading>,
    descended: usize,
    done: bool,
}

/// What a walk reads of the node its path ends at, as it goes: the points
/// themselves, in the node the walk has descended to.
#[derive(Debug)]
enum Reading {
    Points(Vec<Point>),
    Branches(Vec<Branch>),
}

#[derive(Debug, PartialEq)]
pub enum Advance {
    /// The next message: what this read places of the node's children, and
    /// how many children it has where its source knows.
    Children(NodeId, Vec<Run>, Option<Rank>),
    /// The walk needs this node's distribution before it can go on.
    Missing(NodeId),
    Done,
}

impl Walk {
    /// A read below `node`; a part's mass must be a number.
    pub fn new(node: NodeId, Read { path, within }: Read) -> Result<Self> {
        let within = match within {
            Some(Within::Points(parts)) => {
                if parts.iter().any(|part| part.ln_mass.is_nan()) {
                    return Err("a part's ln mass is not a number".into());
                }
                Some(Reading::Points(
                    parts.iter().flat_map(Part::points).collect(),
                ))
            }
            Some(Within::Branches(branches)) => Some(Reading::Branches(branches)),
            None => None,
        };
        Ok(Self {
            node,
            path,
            at: 0,
            within,
            descended: 0,
            done: false,
        })
    }

    pub fn advance(&mut self, tree: &mut Tree, measure: usize) -> Result<Advance> {
        if self.done {
            return Ok(Advance::Done);
        }
        let node = self.node;
        // The path walked, its last branch named this node: it is read only
        // for what is asked within it.
        let step = match (self.path.get(self.at), &mut self.within) {
            (Some(&branch), _) => Ok(branch),
            (None, Some(within)) => Err(within),
            (None, None) => {
                self.done = true;
                return Ok(Advance::Done);
            }
        };
        let Some(distribution) = tree.distribution(node, measure)? else {
            return Ok(Advance::Missing(node));
        };
        let count = distribution.count();
        // What this node places, the child the walk goes on to, if any, and
        // whether it reads that child too.
        let (runs, next, more) = match step {
            Ok(branch) => {
                // A branch of weight zero has no place: the walk ends before it.
                let Some(run) = distribution.place(branch) else {
                    self.done = true;
                    return Ok(Advance::Done);
                };
                self.at += 1;
                let more = self.at < self.path.len() || self.within.is_some();
                (vec![run], Some(branch), more)
            }
            Err(Reading::Branches(branches)) => {
                let mut children: Vec<_> = branches
                    .iter()
                    .filter_map(|&branch| distribution.locate_rank(distribution.rank(branch)?))
                    .collect();
                children.sort_unstable();
                children.dedup();
                (distribution.runs_of(&children), None, false)
            }
            Err(Reading::Points(points)) => {
                let (children, all) = distribution.children_at(points);
                let runs = distribution.runs_of(&children);
                // Into a child holding them all, to tell them apart there:
                // one point is told already, and its child is all it asks.
                let apart = points.iter().any(|point| *point != points[0]);
                match (children.as_slice(), all && apart) {
                    (&[(run, at)], true) if self.descended < MAX_DESCENT => {
                        self.descended += 1;
                        // The same points as the child's own, told as
                        // finely as they were.
                        let child = Item::Child { run, at };
                        let (top, bottom) = distribution.item_edges(child);
                        for point in points.iter_mut() {
                            *point = point.into_child(top, bottom, 0);
                        }
                        let branch = runs[0].branch(0);
                        (runs, Some(branch), true)
                    }
                    _ => (runs, None, false),
                }
            }
        };
        if let Some(branch) = next {
            self.node = tree.child(node, branch)?;
        }
        self.done = !more;
        Ok(Advance::Children(node, runs, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(weights: &[f64]) -> Distribution {
        Distribution::exact(weights)
    }

    /// The points with these masses below them, each a part of its own.
    fn at(below: &[f64]) -> Vec<Part> {
        let part = |below: &f64| Part {
            top: Point::off(below.ln(), true, 0.0, 0.0).unwrap().digits(),
            ln_mass: f64::NEG_INFINITY,
            count: 1,
        };
        below.iter().map(part).collect()
    }

    #[test]
    fn nodes_are_named_by_their_shared_hashes() {
        let mut tree = Tree::new(2);
        let child = tree.child(ROOT, 5).unwrap();
        let grandchild = tree.child(child, 2).unwrap();
        assert_eq!(grandchild, child_hash(child_hash(ROOT, 5), 2));
        assert_eq!(tree.child(ROOT, 5).unwrap(), child);
        assert_eq!(tree.parent(ROOT).unwrap(), None);
        assert_eq!(tree.parent(grandchild).unwrap(), Some(child));
        assert_eq!(tree.branch(grandchild).unwrap(), Some(2));
        assert_eq!(tree.path(grandchild).unwrap(), [5, 2]);
        assert_eq!(
            tree.chain(ROOT, grandchild).unwrap(),
            [ROOT, child, grandchild]
        );
        assert!(tree.chain(grandchild, child).is_err());
        let mut other = Tree::new(1);
        assert_eq!(other.descend(ROOT, &[5, 2]).unwrap(), grandchild);
        assert!(!tree.contains(child_hash(ROOT, 9)));
        // Another parent claiming the same id is a collision, refused.
        tree.insert_node(child_hash(ROOT, 3), Some((ROOT, 4)));
        assert!(tree.child(ROOT, 3).is_err());
    }

    #[test]
    fn entries_place_children_by_p_and_size_them_by_q() {
        let mut tree = Tree::new(2);
        tree.set(ROOT, 0, weights(&[0.25, 0.75])).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let entry = tree.entry(ROOT, lens, 0).unwrap().unwrap();
        assert_eq!(entry.rank, 1);
        assert!((entry.before - 0.75).abs() < 1e-12);
        assert_eq!(entry.nats_q, f64::INFINITY);
        tree.set(ROOT, 1, weights(&[0.5, 0.5])).unwrap();
        let entry = tree.entry(ROOT, lens, 0).unwrap().unwrap();
        assert!((entry.nats_q - 2f64.ln()).abs() < 1e-12);
        assert_eq!(tree.entry(ROOT, lens, 7).unwrap(), None);
        let statistics = tree.statistics(ROOT, lens).unwrap();
        assert!(!statistics.bounded);
        assert!((statistics.cross - 2f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn a_walk_places_a_path_and_finds_children_at_points() {
        let mut tree = Tree::new(1);
        let mut walk = Walk::new(ROOT, Read::path(vec![2, 1])).unwrap();
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Missing(ROOT));
        tree.set(ROOT, 0, weights(&[0.6, 0.3, 0.1])).unwrap();
        let Advance::Children(ROOT, runs, Some(3)) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root places branch 2");
        };
        assert_eq!((runs[0].rank(), runs[0].branches()), (2, &[2][..]));
        let child = child_hash(ROOT, 2);
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Missing(child));
        tree.set(child, 0, weights(&[0.5, 0.5])).unwrap();
        assert!(
            matches!(walk.advance(&mut tree, 0).unwrap(), Advance::Children(node, _, _) if node == child)
        );
        // The path's last branch placed, the walk is done: it lists nothing more.
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);

        // Points across two children list both; points all inside branch 0
        // (0.4, 1] descend into it, rescaled.
        let mut walk = Walk::new(ROOT, Read::points(at(&[0.9, 0.2]))).unwrap();
        let Advance::Children(ROOT, runs, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root lists the children at the points");
        };
        assert_eq!(runs[0].branches(), [0, 1]);
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);
        let mut walk = Walk::new(ROOT, Read::points(at(&[0.9, 0.5]))).unwrap();
        assert!(matches!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Children(ROOT, _, _)
        ));
        assert_eq!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Missing(child_hash(ROOT, 0))
        );

        // One point is its child alone: nothing below it is asked for.
        let mut walk = Walk::new(ROOT, Read::points(at(&[0.9]))).unwrap();
        assert!(matches!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Children(ROOT, _, _)
        ));
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);

        // Branches list just those children; a weightless one is left out.
        let mut walk = Walk::new(ROOT, Read::branches(vec![1, 2, 7])).unwrap();
        let Advance::Children(ROOT, runs, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root lists the branches it has");
        };
        assert_eq!(runs[0].branches(), [1, 2]);
        let part = |ln_mass| Part {
            top: vec![0x80],
            ln_mass,
            count: 4,
        };
        assert!(Walk::new(ROOT, Read::points(vec![part(f64::NAN)])).is_err());
        // Four points evenly through the lower half of the root, where all
        // three branches have a part; a part reaching past the node's bottom
        // has only its points above it, here 0.625 and 0.875.
        for (ln_mass, placed) in [(0.5f64.ln(), &[0, 1, 2][..]), (0.0, &[1][..])] {
            let mut walk = Walk::new(ROOT, Read::points(vec![part(ln_mass)])).unwrap();
            let Advance::Children(ROOT, runs, _) = walk.advance(&mut tree, 0).unwrap() else {
                panic!("the root lists the children the part's points fall in");
            };
            assert_eq!(runs[0].branches(), placed);
        }

        // A path and then points: the path's branch is placed, and the node
        // it reaches is sampled in its own coordinates.
        tree.set(child, 0, weights(&[0.5, 0.5])).unwrap();
        let read = Read {
            path: vec![2],
            within: Some(Within::Points(at(&[0.9, 0.2]))),
        };
        let mut walk = Walk::new(ROOT, read).unwrap();
        let Advance::Children(ROOT, runs, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root places the path's branch");
        };
        assert_eq!(runs[0].branches(), [2]);
        let Advance::Children(node, runs, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the node the path reaches lists the children at the points");
        };
        assert_eq!((node, runs[0].branches()), (child, &[0, 1][..]));
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);
        // Nothing to read is nothing to compute.
        let mut walk = Walk::new(child_hash(ROOT, 9), Read::default()).unwrap();
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);

        // A path through a weightless branch ends before it; an ended node
        // says it has no children.
        tree.set(child, 0, Distribution::from_weights(&[]).unwrap())
            .unwrap();
        let mut walk = Walk::new(child, Read::points(at(&[0.4]))).unwrap();
        assert_eq!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Children(child, vec![], Some(0))
        );
        let mut walk = Walk::new(child, Read::path(vec![0])).unwrap();
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);
    }
}
