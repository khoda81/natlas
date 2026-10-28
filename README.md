# natlas

A web navigator for nested interval trees, built as the visual client for
[natwalk](../natwalk). It knows nothing about tokens or models: a tree is
nodes whose ordered slots partition their interval, each slot carrying one
weight per *measure*.

## The picture

Pick one measure **p** and one measure **q** (by default the same). A slot of
node N is a box

- **height** ∝ p: N's interval split by p, recursively, so a box's height is
  the p-probability of its whole path;
- **width** = −log q nats, starting where N's own box ends, so a box's left
  edge is the q-surprisal of its path.

A node's slot boxes therefore cover N's mass × Σ p·(−log q) in area: entropy
when p = q, cross-entropy otherwise. Comparing p with q = p shows KL(p‖q) as the
extra area. Color repeats the conditional p when p = q and shows ln(p/q) per
slot otherwise, red where q under-weights it.

## The canvas

Depth is unbounded, horizontal scale is unbounded (it is the base of the log),
and vertically the view never leaves the root interval. The view height is
coupled to depth: with the left edge at x nats the view spans e^−x of the
root, so a node at the left edge keeps the same screen height however deep it
is. Scrolling deeper is therefore also zooming in.

Absolute vertical coordinates would lose precision a few dozen nats deep, so
the camera is anchored to a nearby slot boundary instead, and every position is
a same-sign sum of masses outward from it (see `src/layout.ts`). The tests pin
exact geometry 80 halvings deep.

| input | motion |
|---|---|
| wheel, ← → | deeper / shallower, zooming about the pointer |
| drag | grab and move |
| shift + wheel, ↑ ↓ | along the interval |
| ctrl + wheel, pinch, + − | horizontal scale |
| double-click | glide to a box |
| 0 · s · b | root · swap p and q · nats/bits |

## Protocol

`proto/natlas/v1/natlas.proto`. A server streams `ServerMessage`s: one `Hello`
naming the measures and the log scale, then `Run`s, each listing some children
of one node by decreasing weight under measure 0.

- Weights are only relative. Every weight is an integer log against the node's
  first child (1/256 nat in the exporters), so measure 0 travels as unsigned
  gaps between consecutive ranks. The client normalizes each node into its own
  interval; nothing on the wire has to sum to 1.
- The unlisted remainder is one rest per measure, drawn as a tail slot. The
  latest run to reach the end of a node's listing replaces it.
- Nodes are named by 64-bit handles hashed from their path of ranks, so client
  and server both derive them; no ids are assigned.

Clients steer with `Focus`: an anchor and a node budget, explored best first
below the anchor. A `Snapshot` is a recorded stream for a focus on the root;
that is what the client loads today.

## Data

Datasets live in `public/data/` (git-ignored) with a `manifest.json`; a `.pb`
snapshot can also be dropped onto the page.

```bash
bun run data      # Zen of Python character n-grams, no ML needed
cd export
uv run hf.py      # Qwen3-1.7B as p, Qwen3-0.6B as q
uv run hf.py --p HuggingFaceTB/SmolLM2-360M --tokenizer HuggingFaceTB/SmolLM2-360M \
  --q context:4 --q temperature:1.6
```

Both use natwalk's own search to discover the tree, then evaluate the other
measures on the same paths. A q measure is another model with the same
tokenizer, `context:N` (p seeing only its last N tokens) or `temperature:T`.

## Development

[Bun](https://bun.sh) runs everything: install, the dev server (`serve.ts`,
which bundles the page and serves `public/`), tests and the static build.

```bash
bun install
bun run dev     # http://localhost:5173, PORT to change
bun test
bun run build   # static site in dist/, datasets included
bun run proto   # after editing the .proto; needs protoc
```
