# Rendering performance

The 2 ms target is a CPU budget for each render callback, including its synchronous DOM
layout. It has not been achieved for every frame. A 180 Hz display has a 5.56 ms
refresh interval; the tighter 2 ms application budget leaves room for browser
paint, compositing, input, and other work.

## Frame ownership and invalidation

Rust lays out the frame and keeps it. JavaScript receives a packed numeric
array of visible rectangles and the requests the view lacks; picking and
reanchoring query the retained frame and the tree directly.

A node's layout costs one binary search to the first visible child, then one
step per drawn box: children at least a pixel tall, bands about a pixel tall,
and gaps. A band's area-preserving width comes from a per-run suffix sum of
p (−ln p) when p = q, and from a few sampled children otherwise, so a frame
never visits the children hidden inside a band. Picking walks down from the
origin through the laid-out nodes to the one child under the point.

The UI and the fetch pump share geometry only when the tree revision, lens,
camera, and viewport match. A changed anchor remains a changed cache key, even
when it denotes the same interval: rebuilding from that anchor can change
floating-point rounding and edge pixels. The inspector keeps a node's entropy
and cross-entropy, a pass over all its children, until the node, the lens, or
what is known of the node changes.

Frames have explicit lifetimes. The UI retains the displayed frame while a
newer demand frame is prepared, frees superseded frames, and reuses up to two
native and JavaScript buffer sets. All views into a frame become invalid after
`Frame.dispose()`.

Immutable paths and glyphs are cached, and the inspector and top path panel
reuse DOM nodes. Text widths are memoized at the original font size, with a
bounded cache and invalidation on display-scale changes.

## Reproduce the browser benchmark

```bash
bun run bench
```

Open the printed loopback URL and click **Run frame benchmark**. The harness
uses the real `App.render` method and page styles. Its synthetic sources keep
four requests pending, so no model inference or network response is needed.
Run ingestion and fixture construction happen before the timed frames.

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
- `?scenario=none&frames=1&label=pixels`: only settled UI snapshots.
- `?trace=1&frames=300&label=trace`: include per-frame layout, hover, DOM,
  measurement, and canvas-call timings. Tracing itself adds overhead.

Reports are written to `/tmp/natlas-frame-benchmark/<label>.json`, including
canvas PNGs/hashes and the inspector, path, and readout markup. Pixel readback
happens after the measurements because it can change the canvas backend. Each
snapshot renders three times so path-panel resizing and repeated draws
are represented.

`PORT` and `NATLAS_PERF_OUTPUT` override the port and output directory.
`NATLAS_PERF_ROOT` selects a comparison checkout whose tree API matches this
harness; compile that checkout's own Wasm first and make its dependencies
available. A checkout with another API runs its own `bun scripts/bench-frames.ts`
on another `PORT`. Keep baseline and current measurements sequential to avoid
CPU contention.

An optional `NATLAS_PERF_FIXTURE` JSON file containing `{listing, labels}` enables
`?live=1&frames=300&label=model`, which replays a captured root distribution
and label bytes without contacting its model. `listing` holds a complete
distribution heaviest first, `branches` and `lnWeights`; `labels` is the usual
labels JSON array.

## Measured results, 2026-10-31

Headless Chromium on Linux, a 1200 px wide canvas at device scale 1, 15
warm-up frames and 300 measured frames per scenario. The previous commit, run
the same way on the same machine, is the baseline; these numbers are not
comparable with ones taken elsewhere.

| Scenario | Before (median) | After (median) | After (p95) | After, frames ≥ 2 ms |
| --- | ---: | ---: | ---: | ---: |
| 150,000 uniform branches, hover | 13.840 ms | 1.310 ms | 2.020 ms | 17/300 |
| 150,000 uniform branches, moving camera | 15.250 ms | 2.710 ms | 3.725 ms | 261/300 |
| 150,000 long-tail branches, hover | 14.200 ms | 1.485 ms | 2.345 ms | 80/300 |
| 150,000 long-tail branches, moving camera | 21.695 ms | 9.350 ms | 13.060 ms | 298/300 |
| 80-level tree, moving camera | 0.845 ms | 1.055 ms | 1.840 ms | 6/300 |

Layout itself takes about 0.2 ms of the long-tail camera frame; canvas text
drawing takes most of the rest. These timings exclude ingesting runs and
inference.
