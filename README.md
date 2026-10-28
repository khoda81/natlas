# natlas

A web navigator for nested interval trees, built as the visual client for
[natwalk](https://github.com/khoda81/natwalk). It knows nothing about tokens or
models: a tree is nodes whose children partition their interval, with one
weight per child under each *measure*, and children are numbered labels whose
meaning comes separately.

![SmolLM2-360M continuing "The most interesting consequence of information theory is"](docs/text.png)

A language model's tree of texts: each box is a token, as tall as its
probability and as wide as its surprisal, so a node's boxes cover its entropy.

![MuScriptor transcribing a guitar, with a piano roll of the transcription so far](docs/music.png)

[MuScriptor](https://github.com/muscriptor/muscriptor) transcribing a
recording: the boxes are its events (time shifts, here), and the panel above
is a piano roll of the path to the hovered box.

![MuScriptor hearing the recording against MuScriptor hearing silence](docs/music-silence.png)

The same model with q hearing silence instead: red boxes are events the audio
made likelier, and their extra width is what it told the model, KL(p‖q).

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
| ctrl + wheel, pinch, + − | horizontal scale, about the left edge, leaving the vertical view alone |
| double-click | glide until a box stands like the root, its children filling the view |
| hover | show the text down to the box above the canvas; it stays until the next hover |
| 0 · s · b | root · swap p and q · nats/bits |

## Protocol

`proto/natlas/v1/natlas.proto`. One server serves one measure through one
call, `Expand(node, horizon)`, which streams the node's handle and then its
children heaviest first, down to 2^−horizon of its weight, as
`Edge { label, odds }`.

- A child is a 32-bit label; what labels mean (a vocabulary) is agreed
  outside the protocol. Nodes are 64-bit handles the server chooses and the
  client only learns: a request names a node by its parent's handle and its
  label, or names nothing for the root, and the reply starts with the node's
  own handle. A server that has forgotten a handle answers `NOT_FOUND`, and
  the client names that parent again the same way, going up only as far as
  the server has forgotten. Horizon 0 asks for the handle alone.
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

Comparing p with q means asking two servers about the same nodes, each by
its own handles. A `Snapshot` is a recorded server, one `Expand` per node with
the handle it answered; the client replays it, re-encoding it for each
request.

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
fetches its way down on load. `/labels.json` holds each token's bytes. With
`bun run dev` running, the page finds a local server on its own and lists it
first; `?server=URL` points it elsewhere.

The server's handles hash label paths, so they survive a restart, but it
cannot invert them: it remembers the parent and label of each node it has
expanded. After a restart the client names the forgotten ancestors again,
asking for handles alone, which costs round trips but no forward passes.
Models load in float32 by default, which is both more precise and, on a GTX
1660 Ti, about 5× faster than float16; `--dtype` trades that for memory on
larger models.

`--p` also takes a `.gguf` file, which llama.cpp runs with every layer on the
GPU (`--gpu-layers`, and `--n-ctx` for the deepest path it can hold).
llama-cpp-python is built for CUDA on first run, which takes a while. A
request that shares a prefix with the previous one reuses its KV cache, so
walking down a path costs one token per step.

```bash
uv run serve.py "Once upon a time" \
  --p ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/*/Qwen3-4B-Q6_K.gguf \
  --q temperature:1.6
```

### Music

```bash
cd export
uv run scribe.py song.mp3 --at 30 --q silence
```

`export/scribe.py` serves [MuScriptor](https://github.com/muscriptor/muscriptor)'s transcriptions of a
recording from `--at` seconds on. MuScriptor writes music 5 seconds at a time,
as MIDI-like events: a tie prologue naming the notes sustained into the
window, then time shifts, instruments, note-ons and note-offs, pitches, drum
hits and an EOS. A path is one transcription of the whole recording, scrolling
through the windows as `TranscriptionModel.transcribe` does, as in natwalk's
MuScriptor demo: EOS is the model's decision to move on, each later window's
prologue is forced from the notes the path leaves sounding (probability one),
a window at MuScriptor's 2000-event limit gets a forced EOS, and the last
window's EOS ends the path. A path's probability is the one MuScriptor's
chunked decoder gives that transcription.

The measures are the model (`--p small|medium|large`) and each `--q`: another
size hearing the same audio, `silence` for the model hearing nothing (p
against it shows what the audio told the model), `instruments:piano,drums`
for the model told what to expect, or `temperature:T`. Each measure keeps a
few KV caches (`--slots`), one per window in reach, stepped back to the prefix
a request shares with what they hold.

Its `/labels.json` is a notes file rather than an array of bytes:
`{"kind": "notes", "frameRate", "window", "windows", "at", "audio", "events":
[[type, value], …], "instruments", "drums"}`. The client reads its kind and
draws each box as its event: a note head for a note-on and a ring for a
note-off in the instrument's color, a tie, a drum ×, a clock with the time in
the recording for a shift, a stop for an EOS. The panel above the canvas
becomes a piano roll of the path, following the window being written: the
hovered event's notes are outlined and a cursor marks the time written so
far. Drag its corner to make it taller; wheel or drag to scroll pitches and
time, ctrl + wheel to zoom time, double-click to follow again. Hovering a
note names it. Space plays the transcription in view, shift + space plays it
over the recording (`/audio.wav`).

## Data

Datasets live in `public/data/` (git-ignored) with a `manifest.json`: per
dataset, one `.pb` snapshot per measure and a `.labels.json` of label bytes,
base64 in a JSON array. A token can hold part of a character, so the client
decodes a path's bytes together and shows each box as the characters its
token finishes after its ancestors, or as hex where it finishes none.
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

The exporters (`export/hf.py`, `export/synthetic.py`) use a
[natwalk](https://github.com/khoda81/natwalk) checkout beside this one;
the live servers need only what their script headers declare, which `uv run`
installs.

## License

[MIT](LICENSE)
