# Rust core

[Project setup](../README.md) · [Wire protocol](../proto/README.md)

The browser and Python servers share the same [tree](src/tree.rs),
[distribution](src/distribution.rs), and [demand session](src/session.rs)
logic. The core exposes it through [WebAssembly](src/wasm.rs) for the browser
and [PyO3](src/python.rs) for the servers. The browser also runs a session
locally when reading a snapshot.

## Tree and demand

The tree names each node by a hash of its branch path. A distribution records
a node's known children under one measure as runs of consecutive ranks. Each
run carries an exact top edge and log weights for the children in it. The
mass between runs is known as a *gap*. If the source supplies the number of
children with weight, the last rank is known too. See the
[exact runs design notes](../docs/exact-runs.md) for why a run's precision
follows the view.

The session places a requested path first, then prioritizes distributions by
the number of physical pixel rows that wait on them. For stretches of children
thinner than a row, the session samples at about two points per row, evenly
spaced through their mass. The server's [scheduler](../export/README.md#scheduling)
uses these priorities to choose what to compute.

## Anchored geometry

Absolute vertical coordinates would lose precision a few dozen nats deep.
The camera tracks one point, defined in [point.rs](src/point.rs). A path below the origin
leads to a node covering the view, and base-256 digits give the fraction of
that node's $p$-mass above the point. Zooming first anchors the point under
the pointer and then changes depth. Moving the anchor up a node retains its
precision, so zooming out and back in can return to the same place.

An edge is a point too. A run's top has as many digits as the view asks for,
and the later edges add the weights above them within that run. In
[layout](src/layout.rs), each node is placed using its two edges nearest the
anchor. A binary search finds the first child in view. This keeps even a very
light child in its place among its neighbors.

## Rasterization and picking

A physical pixel is a half-open square from its top-left corner. If an edge
falls in it, the pixel shows the outline. A pixel wholly inside the children
shows the fill, and the rest show the background. The edges include the
children's left side, the top and bottom, and steps between widths. Siblings
of equal width meet without a line.

A child or gap one pixel or more tall gets a box that can be labeled and
picked. A gap's width is estimated from its mean weight when $p = q$ and its
count is known, or from adjacent children with known width. A branch whose
$q$ is unknown can reach beyond the view. Picking starts at the shallowest
node laid out and descends to the child whose interval holds the pixel's
top-left corner, even if that child is thinner than a pixel.

The layout tests cover deep anchors, edges beside children of mass
$2^{-3000}$ relative to their parent, and picks beyond the range of a float.
The [rendering checks](../docs/performance.md#check-rendered-pixels) compare
browser pixels with the rule computed from the children's edges.
