# Demand sessions

A design for review: how the browser tells a server what it is looking at,
and who decides what is computed next. It replaces the `Expand` RPC.

## The problem

A read the view lacks passes through five queues and two caches today, each
deciding with part of the picture:

| stage | decides | knows | does not know |
|---|---|---|---|
| layout | what the frame lacks, its visible pixels | what is on screen | what a read costs |
| `Expander` | what to send; 4 compute slots | priority when sending | what the server holds |
| browser | which request gets one of 6 connections | nothing | everything |
| server lock | whose pass runs next | arrival order | priority, staleness |
| core walk | how far a read descends | the tree | that it was sent as one cheap read |
| server and KV caches | what to forget | recent use | what the view needs next |

The client knows priority and staleness but guesses cost (`tree.has` as "the
server holds it", `walk()` always as "computes"). The server knows cost but
sees neither priority nor staleness. Every latency bug so far was two of these
disagreeing, and each fix moved the disagreement: cancelling stale reads,
pausing views during flights, handing the model over per pass (which then
thrashed the KV cache), bypassing the slots for held reads.

## The principle

One place decides, the one that knows both: the server. The client says what
it is looking at, and nothing else, every time the view changes; the server
works out what the view lacks, answers what it holds at once, and spends the
model on the most valuable missing distribution, choosing again after every
forward pass.

The view is always an interval of one node's p mass, the uniform space of the
picture, seen down to some depth. So the client sends that node, the
interval, the pixel rows it covers, and how far right the screen reaches.

The rows are the view's worth, and the server lets it flow down the tree: an
interval of a node splits among the children it covers, each getting the
share of the rows its covered mass takes, a real number. A child worth a row
or more is an interval of its own and goes on, while its children start on
screen. The interval is read at evenly placed points, two a row, each finding
the child that holds it, the next point the first past that child: every
child worth a row is found and goes on, and a stretch of thinner ones, drawn
in the rows they fall in, is sampled and not gone into, in whatever order
the children run:
Monte Carlo sampling of the tree at evenly placed points. Each child's top is
sent to 1/256 of a row of the interval it was found in, and sent again when
a later view asks it more precisely. A missing distribution is worth the
rows waiting on it, so the most valuable is about the most pixels the view
lacks.

A node costs its visible children and a binary search, and only the children
at an interval's two ends are cut by it; nothing is tracked per pixel. What
waits on a missing distribution stays waiting, and when the distribution
arrives only what waited on it reads on: the work is about one forward pass's
worth of bookkeeping, not the view's.

There is then nothing to cancel (a view no longer stated is gone), no slots or
connection limit (one connection), no guess about cost on either side, no
queue but the server's, and nothing for the client to work out: its layout
draws what it has and asks for nothing.

## Protocol

One WebSocket per server, at `/session`, carrying protobuf frames: a page has
one per server, so a socket is a client. `Search` stays a unary Connect call.

A WebSocket rather than a server stream plus a POST per demand: the stream and
the posts would hold two of the browser's six connections and keep a second
way in for requests. Browsers stream a request body only over HTTP/2, which a
local server does not offer, so a Connect bidirectional stream is out.

```proto
// Client to server: what the client is looking at now, in place of what it
// looked at before.
message Demand {
  // The measures in view, by spec: p places children and orders them, q sizes them.
  string p = 1;
  string q = 2;
  optional View view = 3;
  // Placed before anything else: a flight's or a restore's path, or the
  // origin's place in its parent when the view reaches past the origin.
  optional Path path = 4;
}

// The view: an interval of a node's p mass, and the rows it covers.
message View {
  // The deepest node the client knows that holds the whole view.
  fixed64 node = 1;
  // The fraction of the node's mass above the view, as digits in base 256,
  // most significant first: exact however deep the view is.
  bytes top = 2;
  // ln of the view's height, as a fraction of the node's mass.
  double ln_height = 3;
  // The pixel rows the view covers: physical ones, the picture's finest.
  double rows = 4;
  // How far right the screen reaches, in nats of -ln q from where the
  // node's children start.
  double right = 5;
}

message Path {
  fixed64 node = 1;
  repeated uint32 branches = 2;
}

// Server to client, in one frame per batch.
message Updates {
  repeated Update updates = 1;
}

message Update {
  oneof kind {
    // Children of a node, under a measure, the client was not sent before.
    Answer answer = 1;
    // The demand names a node this server has not learned: the client
    // names it again by its path, from a node the server knows.
    fixed64 unknown = 2;
    // The demand's path is placed as far as the source places it.
    Path placed = 3;
    // The source will never have this distribution: a recording that did
    // not record it.
    Missing lacks = 4;
    // Computing a distribution failed, too long for the context or worse.
    Failed failed = 5;
  }
}

message Answer {
  string measure = 1;
  fixed64 node = 2;
  // Its parent and branch, absent at the root: the client may not have
  // named a node the view reached.
  optional fixed64 parent = 3;
  optional uint32 branch = 4;
  Children children = 5;
}

message Missing {
  string measure = 1;
  fixed64 node = 2;
}

message Failed {
  string measure = 1;
  fixed64 node = 2;
  string message = 3;
}
```

`Children`, `Run`, `Snapshot` and `Recording` are unchanged, so recordings
keep their format. `ExpandRequest`, `Points`, `Part`, `Branches` and the
`Expand` RPC go.

## Server

### The session, in the core

A `Session` in `core/src/session.rs` holds one client's demand and what it
has been sent. The Python server and the browser's recordings both use it.

- `demand(demand)` replaces the view and the path, and starts over from the
  view; a path stated again goes on where it was.
- `step(tree)` reads as far as the tree's distributions go and returns what
  is new to the client: the path, node by node, placing each branch under p
  and q; then every interval ready to read. An interval needs p, to split
  among the children, and q, to size them; each child worth a row goes on
  while its children start left of `right`. An interval whose distribution
  is missing waits on it, and reads on in the step after it arrives.
- `blocked()` lists the missing distributions, each worth the rows waiting on
  it, or +inf on the path, which also says the rest of the path below it.
- `lacks(measure, node)` and `failed(measure, node, message)`: the source
  will never have a distribution, or could not compute it. Probes and the path
  stop there, and the client is told once.

What was sent is kept per measure and node as the ranks sent, so a step that
finds nothing new sends nothing. A new session (a reconnect) has sent nothing,
and the client's tree ignores answers it already holds.

### The scheduler

One thread per model owns the core tree, the distribution cache and the
model. Its loop:

1. Take every new demand and every `Search` path to learn since the last
   turn.
2. `step` each session; send what it returns.
3. Take the next client in turn with something blocked; under the measure
   of its most valuable blocked distribution, take a walk from each client
   in turn: a blocked distribution and, where the path goes on below it, the
   next nodes along the path, while the pass has room. A transformers model
   reads 256 positions of one path a pass, getting every position's
   distribution out of it; llama.cpp reads as many tokens, of any paths, as
   the batch it measured at startup. Back to 1.
4. With nothing blocked, wait for the next demand.

A flight's path is computed before anything else, while the view is answered
from the cache in between. A demand that moves on drops what the old view
needed at once.

The socket handler only decodes demands into the scheduler's inbox and sends
the updates it returns. `Turns`, one thread per request, the `names` lock and
the Connect streaming code go; `Search` posts the path to learn to the inbox.

## Client

The core layout computes the demand's view from the camera exactly, as it
places the camera's point: the deepest node covering the view, the view's top
in it as digits, ln of the view's height, its physical pixel rows, the
screen's right edge. The client
sends a demand whenever it changes, and puts every answer into its tree.

A path is stated by a flight, a restore, or the view reaching past the origin
(the origin's branch from its parent); `placed` ends it, and the client checks
its tree to tell placed from not. A `lacks` of the origin is the note that a
recording does not continue there.

`ConnectSource` becomes a WebSocket source that reconnects with the same
demand after a drop. A recording runs the core `Session` over its recorded
tree, all measures in one, answering in the same task and saying `lacks` of
what it did not record.

The layout's requests, the `Expander` and its slots, maps, retries and
cancellation, and the flight pause all go.

## What stays, what goes

Stays: node ids, parts and digits, packed distributions, the cache by bytes,
the KV sequences, `Search`, recordings.

Goes: the `Expand` RPC and positional answers; the layout's requests (gap
sampling, band refinement); the `Expander`; the browser's connection limit;
`Turns`, per-request threads and the tree lock; the flight pause.

Behaviour that changes on purpose: a distribution is computed in priority
order rather than arrival order; nothing off screen is ever computed; several
clients on one server take turns. Thin stretches get two samples a row and
no more, so their estimated width is no longer refined as the horizontal scale
grows; a client that wants that says the view covers more rows.

## Plan

1. `Session` in the core, with tests: worth splitting among children and
   exactly at an interval's ends, thin stretches sampled, the lens, the path
   and what is ahead of it, sent filtering, replacement, `lacks`, `failed`,
   `unknown`.
2. The proto messages; the server's scheduler and WebSocket endpoint (adds
   `websockets` to `serve.py` and `scribe.py`); server tests with two
   clients.
3. The client: the view from the camera in the core layout, the WebSocket
   and recording sources, `main.ts`; tests with a fake socket.
4. Measure on Qwen in the browser: a flight, zooming out to the parents,
   panning across siblings, and a second tab at once.
