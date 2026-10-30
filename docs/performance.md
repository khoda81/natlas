# Rendering performance

The optimization keeps the native canvas drawing commands, fonts, geometry,
branch grouping, picking order, discovery requests, and interactions. The 2 ms
target is a CPU budget for each render callback, including its synchronous DOM
layout. It has not been achieved for every frame. A 180 Hz display has a 5.56 ms
refresh interval; the tighter 2 ms application budget leaves room for browser
paint, compositing, input, and other work.

## Frame ownership and invalidation

Rust retains the complete layout frame. JavaScript receives a packed numeric
array of visible rectangles, while picking and reanchoring query the retained
frame directly. Hidden vocabulary entries no longer make a Rust → JavaScript →
Rust round trip on every render.

Subpixel branches share an immutable slot snapshot and their node metadata.
Their successive accumulated y coordinates are retained in compact numeric
buffers, and picking searches those coordinates. Accumulation order stays the
same: replacing it with prefix-sum subtraction changes rounding at depth and at
merge boundaries. Expanded-child exceptions are snapshotted, so later discovery
cannot mutate an existing frame's selections.

Slots cache log weights, entropy, cross-entropy, and label lookup alongside the
existing probability sums. Compact integer vocabularies use direct lookup;
sparse labels still support the entire unsigned 32-bit range. Scalar coverage,
membership, and statistics queries avoid exporting full listings just to read a
field or update the inspector.

The UI and completion-driven fetch pump share geometry only when the tree
revision, lens, camera, and viewport match. A changed anchor remains a changed
cache key, even when it denotes the same interval: rebuilding from that anchor
can change floating-point rounding and edge pixels. A full request queue does
not recalculate demand until a slot becomes available.

Frames have explicit lifetimes. The UI retains the displayed frame while a
newer demand frame is prepared, frees superseded frames, and reuses up to two
native and JavaScript buffer sets. All views into a frame become invalid after
`Frame.dispose()`. Diagnostic exports such as `frame.known` remain available,
but materialize hidden branches and should stay outside the render path.

Immutable paths and glyphs are cached, and the inspector and top path panel
reuse DOM nodes. Text widths are memoized at the original font size, with a
bounded cache and invalidation on display-scale changes. Every render still
issues the original canvas drawing commands: skipping an apparently identical
draw changes accumulated antialiasing at fractional canvas edges. Bitmap copies
and transformed fonts were also rejected because they changed pixels.

## Reproduce the browser benchmark

```bash
bun run bench
```

Open the printed loopback URL and click **Run frame benchmark**. The harness
uses the real `App.render` method and page styles. Its synthetic sources keep
four requests pending, so no model inference or network response is needed.
Listing ingestion and fixture construction happen before the timed frames.

Each scenario warms up for 15 frames, measures one callback per animation frame,
and forces DOM layout before stopping the timer. Results include all samples,
median, p95, p99, maximum, and the number of callbacks taking at least 2 ms.
They measure synchronous main-thread work, not asynchronous GPU rasterization
or presentation latency. Run on an otherwise idle machine and in the target
browser; Chromium timings do not establish Zen/Firefox timings or sustained
180 Hz presentation.

Useful URL parameters:

- `?frames=300&label=current`: 300 measured frames per scenario.
- `?scenario=long%20tail%20camera&frames=300&label=camera`: one scenario.
- `?scenario=none&frames=1&label=pixels`: only settled UI parity snapshots.
- `?trace=1&frames=300&label=trace`: include per-frame layout, hover, DOM,
  measurement, and canvas-call timings. Tracing itself adds overhead.

Reports are written to `/tmp/natlas-frame-benchmark/<label>.json`, including
canvas PNGs/hashes and the inspector, path, and readout markup. Pixel readback
happens after the measurements because it can change the canvas backend. Each
parity snapshot renders three times so path-panel resizing and repeated draws
are represented.

`PORT` and `NATLAS_PERF_OUTPUT` override the port and output directory.
`NATLAS_PERF_ROOT` selects a comparison checkout; compile that checkout's own
Wasm first and make its dependencies available. The harness can export the
older combined `App` module without running its network startup. Keep baseline
and current measurements sequential to avoid CPU contention.

An optional `NATLAS_PERF_FIXTURE` JSON file containing `{listing, labels}` enables
`?live=1&frames=300&label=model`, which replays the captured root distribution
and label bytes without contacting its model. `listing` uses the decoded
`Fetched` shape, including its horizon; `labels` is the usual labels JSON array.

## Behavior checks

`bun run test` includes 49 fixtures frozen from commit `c03ef09c`, covering empty
trees, distinct measures, partial tails, merging, 150,000 branches, and an
80-level anchor. Hashes cover every geometry field and its order, sampled
cursor selections, reanchoring, and discovery demand. Separate tests retain old
frames across listing updates and buffer reuse, and verify scalar queries
without allowing full vocabulary exports.

The browser harness additionally compares settled canvas pixels and inspector,
path, and readout output with that baseline. These checks cover deterministic
fixtures; they complement interaction checks and profiling in the user's
browser rather than proving every possible tree or browser configuration.

## Measured results, 2026-10-30

Chromium 155 on Linux, a 1280 px wide canvas at device scale 1.25, 15 warm-up
frames and 300 measured frames per scenario, with the native drawing path:

| Scenario | Median | p95 | Frames ≥ 2 ms |
| --- | ---: | ---: | ---: |
| 150,000 uniform branches, hover | 1.140 ms | 1.830 ms | 8/300 |
| 150,000 uniform branches, moving camera | 2.170 ms | 3.370 ms | 182/300 |
| 150,000 long-tail branches, hover | 0.945 ms | 1.625 ms | 4/300 |
| 150,000 long-tail branches, moving camera | 4.565 ms | 6.475 ms | 287/300 |
| 80-level tree, moving camera | 0.540 ms | 1.005 ms | 2/300 |

All three settled browser snapshots matched the baseline exactly for canvas
pixels, inspector markup, path markup, and readout. These timings exclude cold
listing ingestion and inference. They establish the large improvement and the
remaining gap, not an every-frame 2 ms guarantee. Native text drawing remains
a substantial cost in the text-heavy camera case; changing its rendering path
needs a separate decision about added complexity and visual equivalence.
