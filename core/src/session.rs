//! One client's session with a source: what the client is looking at, read
//! through the tree, and what it has already been sent.
//!
//! The view is an interval of one node's p mass, the picture's uniform space,
//! worth the pixel rows it covers, and seen as far right as the screen
//! reaches. Its worth flows down the tree: an interval of a node splits among
//! the children it covers in proportion to the mass of each it covers. A
//! child worth a row or more goes on, as an interval of its own, while its
//! children start on screen; thinner children, which the picture draws as
//! bands, are sampled instead. A distribution is worth what waits on it, so
//! the most valuable missing one is about the most pixel rows the view lacks.
//! A path, a flight's, comes before anything else.
//!
//! Stepping the session reads as far as the tree's distributions go and says
//! what is new to the client. Intervals whose distribution is missing wait;
//! once it arrives, the next step reads on from them alone. A new demand
//! starts over from its view, so whatever the old view waited on and the new
//! one does not is dropped, and whatever both want is found again at once.

use std::collections::{HashMap, HashSet};
use std::mem;
use std::ops::Range;

use crate::distribution::{ln_between, Distribution, Item, Rank, Run};
use crate::point::Point;
use crate::tree::{Branch, Lens, NodeId, Tree};
use crate::Result;

/// Samples per row of worth in a stretch of children each thinner than a
/// row, which the picture draws as bands about a row tall.
const SAMPLES_PER_ROW: f64 = 2.0;

/// What a client is looking at, in place of what it looked at before.
#[derive(Clone, Debug, PartialEq)]
pub struct Demand {
    /// p places children and orders them; q sizes them.
    pub lens: Lens,
    pub view: Option<View>,
    /// Placed before anything else.
    pub path: Option<Path>,
}

/// An interval of a node's p mass, and the pixel rows it covers.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// The deepest node the client knows that holds the whole view.
    pub node: NodeId,
    /// The fraction of the node's mass above the view, in base 256.
    pub top: Vec<u8>,
    /// ln of the view's height, as a fraction of the node's mass.
    pub ln_height: f64,
    /// Pixel rows the view covers.
    pub rows: f64,
    /// How far right the screen reaches, in nats of −ln q from where the
    /// node's children start.
    pub right: f64,
}

/// Branches down from a node.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub node: NodeId,
    pub branches: Vec<Branch>,
}

/// What a session tells its client.
#[derive(Debug, PartialEq)]
pub enum Update {
    /// Children of `node` under `measure` the client was not sent before.
    /// `edge`, its parent and branch, names a node the view reached, which
    /// the client may never have named.
    Answer {
        measure: usize,
        node: NodeId,
        edge: Option<(NodeId, Branch)>,
        runs: Vec<Run>,
        count: Option<Rank>,
    },
    /// The demand names a node the source has not learned.
    Unknown(NodeId),
    /// The demand's path is placed as far as the source places it.
    Placed(Path),
    /// The source will never have this distribution.
    Lacks { measure: usize, node: NodeId },
    /// Computing this distribution failed.
    Failed {
        measure: usize,
        node: NodeId,
        message: String,
    },
}

/// A distribution the demand waits on and what it is worth: the rows waiting
/// on it, or +inf on the path. On the path, `ahead` is the rest of the path
/// below it, whose nodes a model reading several positions in one pass can
/// compute in the same pass.
#[derive(Debug, PartialEq)]
pub struct Blocked {
    pub measure: usize,
    pub node: NodeId,
    pub worth: f64,
    pub ahead: Vec<Branch>,
}

#[derive(Debug, Default)]
pub struct Session {
    lens: Option<Lens>,
    right: f64,
    /// The path's place: the node it has reached and the branches placed.
    path: Option<(Path, NodeId, usize)>,
    /// Intervals to read on from.
    ready: Vec<Interval>,
    /// What waits on each missing distribution.
    waiting: HashMap<(usize, NodeId), Waiting>,
    sent: HashMap<(usize, NodeId), Sent>,
    /// Distributions the source will never have.
    never: HashSet<(usize, NodeId)>,
    /// Nodes said to be unknown, and the path said to be placed.
    unknown: HashSet<NodeId>,
    placed: Option<Path>,
    /// Said by `lacks` and `failed`, before the next step.
    said: Vec<Update>,
}

/// An interval of a node, its worth, and where the node's children start.
#[derive(Clone, Debug)]
struct Interval {
    node: NodeId,
    top: Point,
    ln_mass: f64,
    worth: f64,
    x: f64,
}

#[derive(Debug, Default)]
struct Waiting {
    intervals: Vec<Interval>,
    /// The path waits here, with these branches still to place.
    path: Option<Vec<Branch>>,
}

impl Waiting {
    fn worth(&self) -> f64 {
        if self.path.is_some() {
            return f64::INFINITY;
        }
        self.intervals.iter().map(|interval| interval.worth).sum()
    }
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// What the client is looking at now. A demand that is no number where a
    /// number is wanted, or names measures the tree has not, is refused.
    pub fn demand(&mut self, tree: &Tree, demand: Demand) -> Result<()> {
        let Demand { lens, view, path } = demand;
        if tree.measures() <= lens.p.max(lens.q) {
            return Err(format!("no measure {}", lens.p.max(lens.q)));
        }
        if let Some(view) = &view {
            let numbers = [view.ln_height, view.rows, view.right];
            if numbers.iter().any(|number| number.is_nan()) || view.rows < 0.0 {
                return Err(
                    "a view's height, rows and right edge must be numbers, rows no fewer than none"
                        .into(),
                );
            }
        }
        // A path stated again goes on where it was.
        let kept =
            self.lens == Some(lens) && self.path.as_ref().map(|(kept, ..)| kept) == path.as_ref();
        if !kept {
            self.path = path.map(|path| {
                let node = path.node;
                (path, node, 0)
            });
        }
        self.lens = Some(lens);
        self.waiting.clear();
        self.ready.clear();
        self.right = f64::NEG_INFINITY;
        if let Some(view) = view {
            if self.known(tree, view.node) {
                self.right = view.right;
                self.ready.push(Interval {
                    node: view.node,
                    top: Point::new(&view.top),
                    ln_mass: view.ln_height,
                    worth: view.rows,
                    x: 0.0,
                });
            }
        }
        let start = self.path.as_ref().map(|(path, ..)| path.node);
        if let Some(node) = start {
            if !self.known(tree, node) {
                self.path = None;
            }
        }
        Ok(())
    }

    /// Whether the client may name `node` to this source; said once if not.
    fn known(&mut self, tree: &Tree, node: NodeId) -> bool {
        if tree.contains(node) {
            return true;
        }
        if self.unknown.insert(node) {
            self.said.push(Update::Unknown(node));
        }
        false
    }

    /// Read on as far as the tree's distributions go, the path first, and
    /// say what is new to the client.
    pub fn step(&mut self, tree: &mut Tree) -> Result<Vec<Update>> {
        let mut updates = mem::take(&mut self.said);
        let Some(lens) = self.lens else {
            return Ok(updates);
        };
        // What waited on a distribution that has arrived, or never will.
        let mut arrived = Vec::new();
        for &(measure, node) in self.waiting.keys() {
            if self.never.contains(&(measure, node)) || tree.distribution(node, measure)?.is_some()
            {
                arrived.push((measure, node));
            }
        }
        for key in arrived {
            let waiting = self.waiting.remove(&key).expect("waiting");
            self.ready.extend(waiting.intervals);
        }
        self.read_path(tree, lens, &mut updates)?;
        while let Some(interval) = self.ready.pop() {
            self.read(tree, lens, interval, &mut updates)?;
        }
        Ok(updates)
    }

    /// The distributions the demand waits on, the most valuable first.
    pub fn blocked(&self) -> Vec<Blocked> {
        let mut blocked: Vec<_> = self
            .waiting
            .iter()
            .map(|(&(measure, node), waiting)| Blocked {
                measure,
                node,
                worth: waiting.worth(),
                ahead: waiting.path.clone().unwrap_or_default(),
            })
            .collect();
        blocked.sort_by(|a, b| b.worth.total_cmp(&a.worth));
        blocked
    }

    /// The source will never have this distribution, as a recording that did
    /// not record it.
    pub fn lacks(&mut self, measure: usize, node: NodeId) {
        if self.never.insert((measure, node)) {
            self.said.push(Update::Lacks { measure, node });
        }
    }

    /// Computing this distribution failed.
    pub fn failed(&mut self, measure: usize, node: NodeId, message: &str) {
        if self.never.insert((measure, node)) {
            self.said.push(Update::Failed {
                measure,
                node,
                message: message.to_owned(),
            });
        }
    }

    /// Whether `measure`'s distribution of `node` is there to read. Where
    /// it is missing, `interval` waits on it; where the source will never
    /// have it, nothing does.
    fn held(
        &mut self,
        tree: &Tree,
        measure: usize,
        node: NodeId,
        interval: Option<Interval>,
    ) -> Result<bool> {
        if tree.distribution(node, measure)?.is_some() {
            return Ok(true);
        }
        if !self.never.contains(&(measure, node)) {
            let waiting = self.waiting.entry((measure, node)).or_default();
            waiting.intervals.extend(interval);
        }
        Ok(false)
    }

    /// Down the path from where it was, placing each branch under p and
    /// sizing it under q.
    fn read_path(
        &mut self,
        tree: &mut Tree,
        Lens { p, q }: Lens,
        updates: &mut Vec<Update>,
    ) -> Result<()> {
        let Some((path, mut node, mut at)) = self.path.take() else {
            return Ok(());
        };
        let mut placed = true;
        while let Some(&branch) = path.branches.get(at) {
            let [held_p, held_q] =
                [p, q].map(|measure| tree.distribution(node, measure).map(|d| d.is_some()));
            let (held_p, held_q) = (held_p?, held_q?);
            if !(held_p && held_q) {
                let missing = if held_p { q } else { p };
                if !self.never.contains(&(missing, node)) {
                    let waiting = self.waiting.entry((missing, node)).or_default();
                    waiting.path = Some(path.branches[at..].to_vec());
                    placed = false;
                }
                break;
            }
            let d = tree.distribution(node, p)?.expect("held");
            // A branch of weight zero has no place: the path ends before it.
            let Some(run) = d.place(branch) else {
                break;
            };
            let count = d.count();
            self.answer(tree, p, node, vec![run], count, updates)?;
            if q != p {
                let d = tree.distribution(node, q)?.expect("held");
                let (run, count) = (d.place(branch), d.count());
                self.answer(tree, q, node, run.into_iter().collect(), count, updates)?;
            }
            node = tree.child(node, branch)?;
            at += 1;
        }
        if placed && self.placed.as_ref() != Some(&path) {
            self.placed = Some(path.clone());
            updates.push(Update::Placed(path.clone()));
        }
        self.path = Some((path, node, at));
        Ok(())
    }

    /// Split an interval's worth among the children it covers: send them,
    /// and read on into each worth a row whose children start on screen.
    fn read(
        &mut self,
        tree: &mut Tree,
        Lens { p, q }: Lens,
        interval: Interval,
        updates: &mut Vec<Update>,
    ) -> Result<()> {
        let node = interval.node;
        if !self.held(tree, p, node, Some(interval.clone()))? {
            return Ok(());
        }
        let d = tree.distribution(node, p)?.expect("held");
        let (children, onward) = split(d, &interval);
        let count = d.count();
        let runs = d.runs_of(&children);
        let mut below: Vec<_> = onward
            .into_iter()
            .map(|(child, mut inner)| {
                let r = &d.runs()[child.0];
                inner.x = interval.x - r.ln_p(child.1);
                (r.branch(child.1), inner)
            })
            .collect();
        let branches: Vec<_> = children
            .iter()
            .map(|&(run, at)| d.runs()[run].branch(at))
            .collect();
        self.answer(tree, p, node, runs, count, updates)?;
        if q != p {
            if !self.held(tree, q, node, Some(interval.clone()))? {
                return Ok(());
            }
            let d = tree.distribution(node, q)?.expect("held");
            let mut children: Vec<_> = d
                .ranks(&branches)
                .into_iter()
                .filter_map(|rank| d.locate_rank(rank?))
                .collect();
            children.sort_unstable();
            children.dedup();
            for (branch, inner) in &mut below {
                // A branch q does not rank weighs nothing under q: it reaches
                // past any screen.
                inner.x = interval.x - d.ln_p(*branch).unwrap_or(f64::NEG_INFINITY);
            }
            let (runs, count) = (d.runs_of(&children), d.count());
            self.answer(tree, q, node, runs, count, updates)?;
        }
        for (branch, mut inner) in below {
            if inner.x < self.right {
                inner.node = tree.child(node, branch)?;
                self.ready.push(inner);
            }
        }
        Ok(())
    }

    /// What of a node's children is new to the client, recorded as sent.
    fn answer(
        &mut self,
        tree: &Tree,
        measure: usize,
        node: NodeId,
        runs: Vec<Run>,
        count: Option<Rank>,
        updates: &mut Vec<Update>,
    ) -> Result<()> {
        let sent = self.sent.entry((measure, node)).or_default();
        let mut fresh = Vec::new();
        for run in runs {
            for ranks in sent.take(run.rank()..run.end()) {
                let offset = |rank: Rank| (rank - run.rank()) as usize;
                fresh.push(run.slice(offset(ranks.start), offset(ranks.end)));
            }
        }
        let count = count.filter(|_| !mem::replace(&mut sent.count, true));
        if fresh.is_empty() && count.is_none() {
            return Ok(());
        }
        let edge = match (tree.parent(node)?, tree.branch(node)?) {
            (Some(parent), Some(branch)) => Some((parent, branch)),
            _ => None,
        };
        updates.push(Update::Answer {
            measure,
            node,
            edge,
            runs: fresh,
            count,
        });
        Ok(())
    }
}

/// The children an interval of a node covers that the client needs, in
/// order and once each, and of those, each worth a row with the interval of
/// it covered (its node and x left for the caller).
///
/// Children run heaviest first, so after the first child wholly inside the
/// interval and thinner than a row, every later one is thinner too: from
/// there to the interval's bottom is one stretch, sampled evenly. Only the
/// children at the interval's two ends are cut by it.
fn split(d: &Distribution, interval: &Interval) -> (Vec<Child>, Vec<(Child, Interval)>) {
    let Interval {
        top,
        ln_mass,
        worth,
        ..
    } = interval;
    // The interval's bottom, unless it reaches the node's.
    let bottom = top.moved(true, 1.0, *ln_mass);
    let worth_of = |ln_covered: f64| worth * (ln_covered - ln_mass).exp();
    let (mut children, mut onward) = (Vec::new(), Vec::new());
    let mut item = d.locate_by(|edge| top.below(edge));
    let mut first = true;
    while let Some(found) = item {
        item = d.next(found);
        let (edge_top, edge_bottom) = d.item_edges(found);
        let last = bottom
            .as_ref()
            .is_some_and(|bottom| !bottom.below(edge_bottom));
        // ln of the mass of the item the interval covers.
        let covered = match (first, last) {
            (true, true) => *ln_mass,
            (true, false) => top.to(edge_bottom).1,
            (false, true) => bottom.as_ref().expect("last").to(edge_top).1,
            (false, false) => ln_between(edge_top, edge_bottom),
        };
        let at_first = mem::replace(&mut first, false);
        if covered == f64::NEG_INFINITY {
            if last {
                break;
            }
            continue;
        }
        let Item::Child { run, at } = found else {
            // A gap: nothing to place in it.
            if last {
                break;
            }
            continue;
        };
        let worth = worth_of(covered);
        if worth < 1.0 && !at_first {
            // The stretch of thin children from here down, sampled evenly.
            let stretch = match &bottom {
                Some(bottom) => bottom.to(edge_top).1,
                None => edge_top,
            };
            let samples = (SAMPLES_PER_ROW * worth_of(stretch)).ceil() as usize;
            for sample in 0..samples {
                let share = ((sample as f64 + 0.5) / samples as f64).ln();
                if let Some(Item::Child { run, at }) =
                    d.locate(ln_between(edge_top, stretch + share))
                {
                    if children.last() != Some(&(run, at)) {
                        children.push((run, at));
                    }
                }
            }
            break;
        }
        children.push((run, at));
        if 1.0 <= worth {
            let ln_child = ln_between(edge_top, edge_bottom);
            let inner_top = if at_first {
                top.into_child(edge_top, edge_bottom, 0)
            } else {
                Point::new(&[])
            };
            onward.push((
                (run, at),
                Interval {
                    node: 0,
                    top: inner_top,
                    ln_mass: covered - ln_child,
                    worth,
                    x: 0.0,
                },
            ));
        }
        if last {
            break;
        }
    }
    (children, onward)
}

/// A known child: its run, and its place in the run.
type Child = (usize, usize);

/// What a client has been sent of one node's children under one measure.
#[derive(Debug, Default)]
struct Sent {
    /// Disjoint, in order, never touching.
    ranks: Vec<Range<Rank>>,
    count: bool,
}

impl Sent {
    /// The ranks of `wanted` not sent before, in order, now sent.
    fn take(&mut self, wanted: Range<Rank>) -> Vec<Range<Rank>> {
        let mut fresh = Vec::new();
        let mut at = wanted.start;
        for sent in &self.ranks {
            if wanted.end <= sent.start {
                break;
            }
            if at < sent.start {
                fresh.push(at..sent.start);
            }
            at = at.max(sent.end);
        }
        if at < wanted.end {
            fresh.push(at..wanted.end);
        }
        if wanted.start < wanted.end {
            let first = self.ranks.partition_point(|sent| sent.end < wanted.start);
            let last = self.ranks.partition_point(|sent| sent.start <= wanted.end);
            let merged = self.ranks[first..last].iter().fold(wanted, |all, sent| {
                all.start.min(sent.start)..all.end.max(sent.end)
            });
            self.ranks.splice(first..last, [merged]);
        }
        fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{child_hash, ROOT};

    const P: Lens = Lens { p: 0, q: 0 };

    fn weights() -> Distribution {
        Distribution::exact(&[0.5, 0.3, 0.2])
    }

    /// Weights 0.5, 0.3, 0.2 for branches 0, 1, 2 under every measure, at
    /// these nodes.
    fn tree_with(measures: usize, nodes: &[NodeId]) -> Tree {
        let mut tree = Tree::new(measures);
        for &node in nodes {
            for measure in 0..measures {
                tree.set(node, measure, weights()).unwrap();
            }
        }
        tree
    }

    /// The whole of a node, worth `rows`, seen `right` nats deep.
    fn whole(node: NodeId, rows: f64, right: f64) -> View {
        View {
            node,
            top: Vec::new(),
            ln_height: 0.0,
            rows,
            right,
        }
    }

    fn looking(lens: Lens, view: Option<View>, path: Option<Path>) -> Demand {
        Demand { lens, view, path }
    }

    /// Each answer as measure:node:branches, and the rest by kind.
    fn said(updates: &[Update]) -> Vec<String> {
        updates
            .iter()
            .map(|update| match update {
                Update::Answer {
                    measure,
                    node,
                    runs,
                    ..
                } => {
                    let branches: Vec<_> = runs
                        .iter()
                        .flat_map(|run| run.branches().to_vec())
                        .collect();
                    format!("{measure}:{node:x}:{branches:?}")
                }
                Update::Unknown(node) => format!("unknown {node:x}"),
                Update::Placed(path) => format!("placed {:?}", path.branches),
                Update::Lacks { measure, node } => format!("lacks {measure}:{node:x}"),
                Update::Failed { measure, node, .. } => format!("failed {measure}:{node:x}"),
            })
            .collect()
    }

    /// What waits, as measure, node and worth to a hundredth.
    fn worths(session: &Session) -> Vec<(usize, NodeId, f64)> {
        session
            .blocked()
            .iter()
            .map(|blocked| {
                let worth = (blocked.worth * 100.0).round() / 100.0;
                (blocked.measure, blocked.node, worth)
            })
            .collect()
    }

    #[test]
    fn worth_splits_among_children_and_goes_on_while_they_start_on_screen() {
        let mut tree = tree_with(1, &[ROOT]);
        let mut session = Session::new();
        let child = |branch| child_hash(ROOT, branch);
        // Ten rows: five on branch 0, three on 1, two on 2.
        session
            .demand(&tree, looking(P, Some(whole(ROOT, 10.0, 5.0)), None))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["0:0:[0, 1, 2]"]);
        assert_eq!(
            worths(&session),
            [(0, child(0), 5.0), (0, child(1), 3.0), (0, child(2), 2.0)]
        );
        // Only branch 0's children start left of 1 nat (−ln 0.5 = 0.69).
        session
            .demand(&tree, looking(P, Some(whole(ROOT, 10.0, 1.0)), None))
            .unwrap();
        assert_eq!(session.step(&mut tree).unwrap(), []);
        assert_eq!(worths(&session), [(0, child(0), 5.0)]);
        // Once it arrives, the next step reads on from there alone.
        tree.set(child(0), 0, weights()).unwrap();
        let updates = session.step(&mut tree).unwrap();
        assert!(matches!(
            &updates[..],
            [Update::Answer { node, edge: Some((ROOT, 0)), .. }] if *node == child(0)
        ));
        assert_eq!(said(&updates), [format!("0:{:x}:[0, 1, 2]", child(0))]);
        assert!(session.blocked().is_empty());
        // Half the root from its top, four rows: all on branch 0, whose
        // children take 2, 1.2 and 0.8 of them. The last is thinner than a
        // row: sampled, not read on into.
        let half = View {
            ln_height: 0.5f64.ln(),
            ..whole(ROOT, 4.0, 3.0)
        };
        session.demand(&tree, looking(P, Some(half), None)).unwrap();
        session.step(&mut tree).unwrap();
        let grandchild = |branch| child_hash(child(0), branch);
        assert_eq!(
            worths(&session),
            [(0, grandchild(0), 2.0), (0, grandchild(1), 1.2)]
        );
    }

    #[test]
    fn an_interval_splits_exactly_at_its_ends() {
        let tree = tree_with(1, &[ROOT]);
        let d = tree.distribution(ROOT, 0).unwrap().unwrap();
        // From 0.4 down 0.25 of the root: 0.1 of branch 0 and 0.15 of 1.
        let interval = Interval {
            node: ROOT,
            top: Point::off(0.6f64.ln(), true, 0.0, 0.0).unwrap(),
            ln_mass: 0.25f64.ln(),
            worth: 10.0,
            x: 0.0,
        };
        let (children, onward) = split(d, &interval);
        assert_eq!(children, [(0, 0), (0, 1)]);
        let near = |actual: f64, expected: f64| {
            assert!((actual - expected).abs() < 1e-12, "{actual} {expected}")
        };
        near(onward[0].1.worth, 4.0);
        near(onward[1].1.worth, 6.0);
        // The first covers branch 0's last fifth, the second branch 1's
        // first half.
        near(onward[0].1.ln_mass, 0.2f64.ln());
        near(onward[1].1.ln_mass, 0.5f64.ln());
        // Their tops: 0.8 of the way into branch 0, and branch 1's own.
        near(onward[0].1.top.to(0.0).1, 0.8f64.ln());
        assert_eq!(onward[1].1.top.to(0.0).1, f64::NEG_INFINITY);
    }

    #[test]
    fn a_thin_stretch_is_sampled_and_not_read_on_into() {
        // A heavy child, then 100 thin ones.
        let mut weights_ = vec![0.5];
        weights_.extend([0.005; 100]);
        let mut tree = Tree::new(1);
        tree.set(ROOT, 0, Distribution::exact(&weights_)).unwrap();
        let mut session = Session::new();
        // 10.2 rows: 5.1 on the heavy child, about a twentieth of a row on
        // each thin one, two samples a row among them: 11.
        session
            .demand(&tree, looking(P, Some(whole(ROOT, 10.2, 100.0)), None))
            .unwrap();
        let updates = session.step(&mut tree).unwrap();
        let Update::Answer { runs, .. } = &updates[0] else {
            panic!("an answer");
        };
        let placed: usize = runs.iter().map(Run::len).sum();
        assert_eq!(placed, 12);
        assert_eq!(worths(&session), [(0, child_hash(ROOT, 0), 5.1)]);
    }

    #[test]
    fn q_sizes_what_p_places_and_both_are_waited_on() {
        let mut tree = Tree::new(2);
        tree.set(ROOT, 0, weights()).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let mut session = Session::new();
        session
            .demand(&tree, looking(lens, Some(whole(ROOT, 10.0, 5.0)), None))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["0:0:[0, 1, 2]"]);
        assert_eq!(worths(&session), [(1, ROOT, 10.0)]);
        // Under q, branch 2 is heavy: its children start at −ln 0.9 < 1.
        tree.set(ROOT, 1, Distribution::exact(&[0.05, 0.05, 0.9]))
            .unwrap();
        session
            .demand(&tree, looking(lens, Some(whole(ROOT, 10.0, 1.0)), None))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["1:0:[2, 0, 1]"]);
        assert_eq!(worths(&session), [(0, child_hash(ROOT, 2), 2.0)]);
        let missing = looking(Lens { p: 0, q: 2 }, None, None);
        assert!(Session::new().demand(&tree, missing).is_err());
    }

    #[test]
    fn a_path_comes_first_goes_on_where_it_was_and_is_said_placed_once() {
        let mut tree = tree_with(1, &[ROOT]);
        let mut session = Session::new();
        let path = Path {
            node: ROOT,
            branches: vec![1, 2, 0],
        };
        let flying = looking(P, Some(whole(ROOT, 10.0, 0.1)), Some(path));
        session.demand(&tree, flying.clone()).unwrap();
        assert_eq!(
            said(&session.step(&mut tree).unwrap()),
            ["0:0:[1]", "0:0:[0, 2]"]
        );
        let middle = child_hash(ROOT, 1);
        let blocked = &session.blocked()[0];
        assert_eq!(
            (blocked.node, blocked.worth, &blocked.ahead[..]),
            (middle, f64::INFINITY, &[2, 0][..])
        );
        tree.set(middle, 0, weights()).unwrap();
        let last = tree.child(middle, 2).unwrap();
        tree.set(last, 0, weights()).unwrap();
        // Stated again, the path goes on from the middle.
        session.demand(&tree, flying).unwrap();
        assert_eq!(
            said(&session.step(&mut tree).unwrap()),
            [
                format!("0:{middle:x}:[2]"),
                format!("0:{last:x}:[0]"),
                "placed [1, 2, 0]".into()
            ]
        );
        assert_eq!(session.step(&mut tree).unwrap(), []);
        // A branch of weight zero ends a path as far as it goes.
        let ended = Path {
            node: ROOT,
            branches: vec![7],
        };
        session
            .demand(&tree, looking(P, None, Some(ended)))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["placed [7]"]);
    }

    #[test]
    fn a_client_is_sent_each_child_once() {
        let mut tree = tree_with(1, &[ROOT]);
        let mut session = Session::new();
        let quarter = |top: u8| View {
            top: vec![top],
            ln_height: 0.25f64.ln(),
            ..whole(ROOT, 4.0, 0.1)
        };
        // The first quarter: branch 0 alone.
        session
            .demand(&tree, looking(P, Some(quarter(0)), None))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["0:0:[0]"]);
        // The quarter from 0.4: branches 0 and 1, of which 1 is new.
        session
            .demand(&tree, looking(P, Some(quarter(0x66)), None))
            .unwrap();
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["0:0:[1]"]);
        session
            .demand(&tree, looking(P, Some(quarter(0x66)), None))
            .unwrap();
        assert_eq!(session.step(&mut tree).unwrap(), []);
    }

    #[test]
    fn unknown_nodes_and_distributions_never_had_are_said_once() {
        let mut tree = tree_with(1, &[]);
        let mut session = Session::new();
        let stranger = child_hash(ROOT, 7);
        session
            .demand(&tree, looking(P, Some(whole(stranger, 4.0, 1.0)), None))
            .unwrap();
        assert_eq!(
            said(&session.step(&mut tree).unwrap()),
            [format!("unknown {stranger:x}")]
        );
        session
            .demand(&tree, looking(P, Some(whole(stranger, 4.0, 1.0)), None))
            .unwrap();
        assert_eq!(session.step(&mut tree).unwrap(), []);
        session
            .demand(&tree, looking(P, Some(whole(ROOT, 4.0, 1.0)), None))
            .unwrap();
        session.step(&mut tree).unwrap();
        assert_eq!(worths(&session), [(0, ROOT, 4.0)]);
        // A recording without the root: nothing waits on it any more.
        session.lacks(0, ROOT);
        session.lacks(0, ROOT);
        assert_eq!(said(&session.step(&mut tree).unwrap()), ["lacks 0:0"]);
        assert!(session.blocked().is_empty());
        let child = child_hash(ROOT, 0);
        session.failed(0, child, "too long for the model's context");
        assert_eq!(
            said(&session.step(&mut tree).unwrap()),
            [format!("failed 0:{child:x}")]
        );
        let nan = View {
            right: f64::NAN,
            ..whole(ROOT, 4.0, 1.0)
        };
        assert!(session.demand(&tree, looking(P, Some(nan), None)).is_err());
    }

    #[test]
    // Lists of rank ranges, some of one range.
    #[allow(clippy::single_range_in_vec_init)]
    fn sent_ranks_merge_and_split_what_is_new() {
        let mut sent = Sent::default();
        assert_eq!(sent.take(5..8), [5..8]);
        assert_eq!(sent.take(0..2), [0..2]);
        assert_eq!(sent.take(0..10), [2..5, 8..10]);
        assert_eq!(sent.ranks, [0..10]);
        assert_eq!(sent.take(3..4), []);
        assert_eq!(sent.take(10..10), []);
        assert_eq!(sent.take(12..13), [12..13]);
        assert_eq!(sent.ranks, [0..10, 12..13]);
    }
}
