# Protocol

[Project setup](../README.md) · [Schema](natlas/natlas.proto)

The browser and servers in this repository share one Protobuf schema.
A distribution is over sequences of unsigned 32-bit integers. The empty
sequence is the root, and its extensions are branches. A server assigns
branch meanings outside the protocol. Each node has $2^{32}$ possible
branches, with zero weight for branches absent from the source distribution.

## Nodes and runs

A 64-bit node ID is derived from its parent ID and branch using the SplitMix64
finalizer. The root ID is zero. The client and server can name a known node
with its ID even deep into the tree. If the server does not know an ID,
it sends an `unknown` update and the client sends the node's path from the root.

A source ranks a node's children in its own order. The language-model server
defaults to heaviest first with ties by increasing branch, and accepts
`--order branch`. The music server always uses branch order. A `Run` carries
consecutive ranks, with the branches, natural log weights, and a top edge.
The top is the fraction of the node's mass above the run, encoded as base-256
digits. Later edges add the weights from that top. If two runs specify the
same edge, the more precise top wins, with the later update breaking a tie.
`Children.count` gives the total number of ranked children when known.

## Sessions

The browser opens one WebSocket per live server at `/session`. It sends a
`Demand` naming $p$ and $q$ by spec, an optional view, and an optional path
to place. The view describes an interval of $p$-mass using a node ID, a
base-256 top, a log height, and its screen coverage. The server places the
path first, then streams batches of `Updates` with the available runs and
the progress or failure of requested work. For display, $p$ orders the
children and splits the height. The client reads $q$ by branch for their
widths. See the [core session logic](../core/src/session.rs) and the
[demand session design notes](../docs/sessions.md).

The unary Connect `Search` RPC at `/natlas.Natlas/Search` maps text to a
branch path and teaches the server each ID along it. Music services return
`unimplemented` for text search.

## Snapshots

A `Snapshot` records the available children of each saved node, identified
by its complete path from the root. The browser answers from these recordings
using the same session logic as a live server. See
[snapshot generation and migration](../export/README.md#generate-snapshots).

Run `bun run proto` from the repository root to regenerate the TypeScript
and Python bindings with [gen-proto.sh](../scripts/gen-proto.sh).
