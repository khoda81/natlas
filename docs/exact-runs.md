# Exact runs

Design for answers whose edges are as precise as the view asks, so that a
server may order a node's children however it likes: heaviest first, as
now, or by branch, with any other order being a relabelling of branches that
server and client agree on outside the protocol.

## Why edges change

A run's edges are f64s today: ln of the node's mass below each edge. That
is precise only relative to the mass below. Heaviest first, the mass below a
child is at most the number of children below it times its own weight, so
every child keeps its extent however light it is. In any other order a
light child can sit above heavy ones: a pitch at p = 1e-20 above notes
holding half the mass has edges ln(0.5 + 1e-20) and ln(0.5), the same f64.
It has no extent at any zoom, and no answer could give it one, because the
format cannot hold it. The server's own edges (`Packed`, f64 sums from the
bottom) have the same floor.

A demand already names its view exactly (`View.top` as base-256 digits,
`ln_height`), and `Distribution::insert` already lets an answer replace what
it contradicts. What is missing is an answer whose precision follows the
view.

## Protocol

```proto
// Consecutive ranks of a node's children in the measure's order.
message Run {
  uint32 rank = 1;
  repeated uint32 branches = 2;
  // ln of each child's weight, conditional on the node.
  repeated double ln_weights = 3;
  // The fraction of the node's mass above the run, as base-256 digits,
  // floored: as precise as the view asks. Each next edge is the one
  // before plus its child's weight.
  bytes top = 4;
}
```

The only change is `ln_below` giving way to `ln_weights` and `top`. A
child's weight is its size, at relative precision, wherever it sits: −ln q
for its width, ln(p/q) for its colour, and its share of a band. Its place
is the run's top plus the weights before it. Nothing is stated twice.

**Precision follows the view.** `top` is a fixed-point fraction, like
`View.top`: outside heaviest-first order rank says nothing about size, so
uniform absolute precision is the right density. It has as many digits as
the view's top at its height, plus one past what places a physical row,
`⌈log₂₅₆(rows)⌉ + 1`: an edge 1/256 of a row exact. A tighter view gets more
digits; there is no floor.

**The server anchors runs where precision needs it.** f64 sums from a run's
top are precise to 2⁻⁵³ of the sum, so the server ends a run, and starts the
next at an exact top of its own, wherever the sum would carry an edge past
the last digit of `top`. Inside a view the sums are no larger than the
view, far finer than its digits; a run breaks only at a heavy child the
view cuts. Heaviest first is the same rule: today's sums from the light end
become runs anchored near the light children a view looks at.

**The most precise run wins.** Edge r is the top of rank r. Where answers
overlap, each edge comes from the run with the most digits that gives it,
and at one precision the latest answer wins: finer replaces coarser.

A `Path` placed for a flight or restore answers each node on it with a run
around its chain child, its top as precise as the chain child's own
extent: the view the camera will have inside it.

## Server

A measure computes ln weights (f32 for models, f64 in recordings). The
server fixes each child's weight once as a dyadic, `w = round(e^ln w)`, the
rounding `point.rs`'s `Dyadic::exp` already defines. The node's mass
`W = Σ w` and every edge's fraction above, `Σ_{i<r} w_i / W`, are exact
rationals, and `top` is one of them floored.

That is the only exact arithmetic: once per run, a sum as large as the node
taken to the run's precision. `Packed` keeps what it keeps now (branches
and ln weights, the steps between weights signed so any order packs) and an
exact sum every 256 children, from which a run's top is a few hundred
fixed-point additions away (measured below).

`Sent` remembers, per rank, the precision it sent, and sends a run again
when a demand needs it more precise.

## Client

A run keeps its `top` digits and its weights. The camera's point shares a
run's leading digits wherever the run is in view, so placing a run is small
arithmetic on the digits past the shared ones, then f64 sums of weights
from there. Layout converts to view heights once per run; `point.rs`
compares points and tops digit by digit, and its rounding of ln edges goes.
Gap masses, worth and band heights are differences of edges.

## Recordings

A snapshot is a server's answers saved to a file, read without a server:
the static datasets in `public/data`. Nothing refines them, so each
recorded run's top holds the digits its smallest recorded child needs.
Existing snapshots are re-exported.

## Order

With exact runs, order is a server's choice per measure: heaviest first, or
increasing branch (`serve.py --order`, scribe by branch: MuScriptor's
vocabulary is already musical, time shifts and pitches ascending). The
session's `split` stops assuming that past the first thin child every child
is thin: it walks points evenly through the interval, two per row as now,
sends each point's child, and jumps to the first point past that child's
bottom. Every child worth a row is hit and goes on; thin stretches are
sampled as now. Layout already walks edges without assuming an order; gap
widths and band colours, estimated from neighbours, get noisier in branch
order but stay correct.

## Exact sums: measured

Five rows of Qwen3-4B's next-token weights (151,936 children; the root
after EOS, entropy 11 nats, and four contexts from 0.01 to 0.85 nats, the
lightest child near e⁻⁴⁰), each in branch order and heaviest first, on one
CPU core. Exact sums are fixed-point over the rows' range: four 64-bit
words. "Per run" is one `top`: the exact sum above a rank and its floored
digits (4 or 12), checked equal both ways and against f64 sums.

| exact sums | per run | memory | build per node |
|---|---|---|---|
| on demand, from the nearer end | 110–130 µs | none | none |
| kept every 16 children | 0.3–0.5 µs | +2.0 bytes/child | +0.5–0.8 ms |
| kept every 64 | 0.4–0.5 µs | +0.5 bytes/child | +0.7 ms |
| kept every 256 | 0.6–0.7 µs | +0.13 bytes/child | +0.7 ms |
| kept every 1024 | 1.7 µs | +0.03 bytes/child | +0.7 ms |
| today's `from_weights`, for scale | | 4.1 bytes/child | 10–11 ms |

Order makes no difference, and the digits themselves take 0.1–0.3 µs.
Summing on demand costs a demand that answers a few hundred runs tens of
milliseconds; keeping a sum every 256 children costs 3% more memory per
node and keeps a run under a microsecond. `Packed` keeps an exact sum
every 256 children: every 16th of its blocks.
