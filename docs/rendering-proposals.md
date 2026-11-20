# Rendering proposals

These are design proposals only; neither feature is implemented by this change.

## Rebalance around a dominant branch

### Problem

A measure q can assign almost all probability to one branch. Its own width,
`-log q`, is then close to zero, making that branch difficult to see or select.
For example, q = 0.999999 gives a width of about 0.000001 nats. Low-probability
alternatives have larger surprisal widths; when p is also concentrated, their
heights can instead be tiny. Unbounded horizontal scaling helps, but a useful
view may still need to regroup around the near-certain sequence.

### Proposal

Offer a view transformation centered on a selected dominant sequence. Re-express
the distribution at each affected node as two branches:

1. the branch containing the selected sequence;
2. everything else, represented as a complementary distribution.

Apply this recursively along the selected sequence. The complementary branch
must retain enough probability mass and conditional structure to expand and
inspect the alternatives; it must not become a purely decorative tail.

The transformation is a lens over the same integer-sequence distribution, not
a mutation of the cached tree. The selected sequence and transformation state
belong to the view. Node identity remains the original label path, and p/q
measure semantics must remain explicit.

### Open questions

- Should the complementary distribution use q, p, or a separately selected
  measure for its horizontal geometry?
- How should the view indicate that its widths no longer use the ordinary
  global `-log q` scale?
- Which camera and ruler operations remain meaningful while the transform is
  active?

### Acceptance criteria

- The near-certain sequence remains visible and selectable despite its tiny
  ordinary surprisal width, and its alternatives remain inspectable.
- Expanding a complementary branch reveals its original child paths.
- Switching the transform off restores exactly the previous geometry.
- Deep paths retain the core's anchored-coordinate precision guarantees.
- Tests compare probability mass and path identity against the untransformed
  distribution.

## Vertical arithmetic/radix ruler

### Problem

The horizontal ruler explains surprisal, but the vertical coordinate has no
corresponding readout. Vertical positions are arithmetic-coded probability
intervals, and inspecting their digits is useful when navigating a sequence
distribution.

### Proposal

Add a vertical ruler beside the canvas. At each tick, show the vertical
position's arithmetic-coded value in a configurable radix. Base 2 and base 16
are useful defaults; the control should accept other integer bases where the
representation remains practical.

The ruler must derive positions from the same anchored camera geometry as the
boxes. It must not convert deep positions through an absolute root-relative
floating-point coordinate, which would lose precisely the digits the ruler is
meant to expose.

### Open questions

- Should ticks be aligned to radix boundaries, ordinary decimal screen steps,
  or a mixture chosen for readability?
- How many fractional digits should be shown at different zoom levels?
- Should the ruler display the interval covered by each tick pixel rather than
  only its representative position?

### Acceptance criteria

- The ruler stays aligned with boxes through pan, zoom, resize, and re-anchoring.
- Radix is configurable and its digits update without changing the camera.
- Deep-view tests verify digits beyond the range of absolute `f64` root
  coordinates.
- Labels remain legible in narrow and short viewports, with a way to hide the
  ruler.
