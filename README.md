# natlas

A web navigator for nested interval trees, built as the visual client for
[natwalk](../natwalk). It knows nothing about tokens or models: a tree is
nodes whose children partition their interval, with one weight per child under
each *measure*, and children are numbered labels whose text comes separately.

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
| double-click | glide until a box stands like the root, its children filling the view |
| hover | show the text down to the box above the canvas; it stays until the next hover |
| 0 · s · b | root · swap p and q · nats/bits |

## Protocol

`proto/natlas/v1/natlas.proto`. One server serves one measure through one
call, `Expand(node, horizon)`, which streams the node's children heaviest
first, down to 2^−horizon of its weight, as `Edge { label, odds }`.

- A child is a 32-bit label; what labels mean (a vocabulary) is agreed
  outside the protocol. Nodes are 64-bit handles hashed from their label path,
  root 0, so the client names any node it has seen listed.
- Each weight is a fraction g of the previous child's, sent as the log-odds of
  g in steps of horizon·ln 2 / 2³¹ nat. Listings are sorted by construction and
  every prefix is valid: the client sums nothing up front, and whatever the
  listed children leave is the rest. One horizon sets both where the server
  stops and how precise the weights are.
- The client asks for what the view needs: a box's children down to a pixel
  under p and, when q differs, every child that could end short of the right
  edge under q. Children q has not listed have an upper bound on q, so their
  width is a lower bound; the tail is drawn the same way, at least as wide as
  the horizon.

Comparing p with q means asking two servers about the same handles. A
`Snapshot` is a recorded server, one `Expand` per node; that is what the client
loads today, re-encoding it for each request.

## Live

```bash
bun run live      # SmolLM2-360M on the GPU, at http://localhost:8421
cd export
uv run serve.py "Once upon a time" --p HuggingFaceTB/SmolLM2-360M \
  --q context:4 --q temperature:1.6
```

`export/serve.py` loads the model and serves each measure as its own `Natlas`
service, `/0/` for the model and `/1/`, `/2/`, … for the `--q` measures, over
the Connect protocol. Its root is the empty text, the start-of-text token
alone, and every text is a node like any other: the prompt is only where the
client starts, a path of token labels in `GET /manifest.json` that the client
fetches its way down on load. `/labels.json` holds each token's text. With
`bun run dev` running, the page finds a local server on its own and lists it
first; `?server=URL` points it elsewhere.

The server learns paths from its own listings, since handles cannot be
inverted. After a restart the client teaches it again, expanding each
ancestor of a node it asks about. Models load in float32 by default, which is
both more precise and, on a GTX 1660 Ti, about 5× faster than float16;
`--dtype` trades that for memory on larger models.

## Data

Datasets live in `public/data/` (git-ignored) with a `manifest.json`: per
dataset, one `.pb` snapshot per measure and a `.labels.json` of label texts.
Snapshots, with or without a labels file, can also be dropped onto the page.

```bash
bun run data      # Zen of Python character n-grams, no ML needed
cd export
uv run hf.py      # Qwen3-1.7B as p, Qwen3-0.6B as q
uv run hf.py --p HuggingFaceTB/SmolLM2-360M --tokenizer HuggingFaceTB/SmolLM2-360M \
  --q context:4 --q temperature:1.6
```

Both use natwalk's own search to discover the tree, then evaluate the other
measures on the same paths, each listed in its own order down to the bit
below its coverage. `hf.py` roots the tree at the empty text too: it records
the path down to the prompt, where natwalk's search begins, and the page opens
there. A q measure is another model with the same
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
