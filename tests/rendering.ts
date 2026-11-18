// Real Canvas pixel checks, served by scripts/check-rendering.ts.
import { layout, pick, top, type Box, type Frame } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { Palette } from "../src/ui/color";
import { draw, NATS, RULER_HEIGHT } from "../src/ui/render";
import { answer, digits, run } from "./build";

const view = { width: 270, height: 340 };
const palette = Palette.current();
const state = { hover: null, ancestors: new Set<string>(), units: NATS, glyph: () => ({ text: "" }), child: (node: bigint, branch: number) => (node << 20n) + BigInt(branch) + 1n, note: null, timeline: null, known: () => [] };
const cam = (phase = 0) => ({ origin: ROOT, x: 0, k: 23, anchor: top(), a: phase / view.height });
const rgb = (color: string) => color.startsWith("#")
  ? color.slice(1).match(/../g)!.map(c => Number.parseInt(c, 16))
  : color.match(/\d+/g)!.map(Number);
const canvas = (dpr: number) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(view.width * dpr);
  canvas.height = Math.round((view.height + RULER_HEIGHT) * dpr);
  canvas.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);
  return canvas;
};
const count = 10000;
const uniform = new IntervalTree(2);
uniform.insert(ROOT, 0, answer(Array(count).fill(1 / count), undefined, 0, 0));
uniform.insert(ROOT, 1, answer(Array(count).fill(1 / count), undefined, 0, 0));
const varying = new IntervalTree(2);
const weights = Array.from({ length: count }, (_, i) => 1 + 0.9 * Math.cos(i / (count - 1) * Math.PI));
const total = weights.reduce((a, b) => a + b, 0);
const p = weights.map(w => w / total);
varying.insert(ROOT, 0, answer(p, undefined, 0, 0));
varying.insert(ROOT, 1, answer(p, p.map((_, i) => count - 1 - i), 0, 0));
// Children about 2 and 1.4 CSS pixels tall, colored apart by ln(p/q).
const thin = new IntervalTree(2);
const comparison = { p: 0, q: 1 };
thin.insert(ROOT, 0, answer([0.6, 0.39, 0.006, 0.004], undefined, 0, 0));
thin.insert(ROOT, 1, answer([0.6, 0.39, 0.008, 0.002], [0, 1, 3, 2], 0, 0));
// A heavy child ending mid-pixel at every ratio, then children thinner than
// a pixel.
const midPixel = new IntervalTree(1);
midPixel.insert(ROOT, 0, answer([0.5, 0.4983, 0.0007, 0.0006, 0.0004], undefined, 0, 0));
// Known children alternate with one-child gaps, so the tail spans runs.
const gapped = new IntervalTree(1);
gapped.insert(ROOT, 0, { runs: Array.from({ length: count / 2 }, (_, i) =>
  run([1 / count], [2 * i], 2 * i, 1 - (2 * i + 1) / count)) });

const results: { name: string; dpr: number; phase: number; error: number }[] = [];
for (const dpr of [1, 1.25, 1.5, 2]) {
  for (const [name, tree, lens] of [
    ["uniform tail", uniform, { p: 0, q: 0 }],
    ["neutral comparison", uniform, { p: 0, q: 1 }],
    ["varying comparison", varying, { p: 0, q: 1 }],
    ["gapped tail", gapped, { p: 0, q: 0 }],
  ] as const) {
    for (const phase of [0, 0.2, 0.61]) {
      const camera = cam(phase);
      const frame = layout(tree, lens, camera, view, { rowPx: 1 / dpr });
      const ctx = canvas(dpr).getContext("2d")!;
      draw(ctx, frame, camera, view, lens, palette, state);
      const pixels = ctx.getImageData(Math.round(10 * dpr), 0, 1, Math.floor(view.height * dpr)).data;
      let index = 0, error = 0;
      for (let row = 2; row < view.height * dpr - 2; row++) {
        const top = row / dpr / view.height, bottom = (row + 1) / dpr / view.height;
        while (index + 1 < frame.boxes.length && frame.boxes[index]!.y1 <= top) index++;
        const colors: number[][] = [];
        for (let i = index; i < frame.boxes.length && frame.boxes[i]!.y0 < bottom; i++) {
          const box = frame.boxes[i]!;
          colors.push(rgb(palette.tile(box.p, box.q, lens.p === lens.q).fill));
        }
        // This column is fully covered. A pixel straddling colors may mix
        // them, but must not include the background. Allow one channel step
        // for 8-bit rounding/dithering; constant tails must stay constant.
        for (let channel = 0; channel < 3; channel++) {
          const values = colors.map(c => c[channel]!);
          error = Math.max(error, Math.min(...values) - pixels[row * 4 + channel]!,
            pixels[row * 4 + channel]! - Math.max(...values));
        }
      }
      results.push({ name, dpr, phase, error });
      frame.dispose();
    }
  }

  for (const phase of [0, 0.2, 0.61, 0.99]) {
    const right = (160 + phase) / dpr, bottom = (80 + phase) / dpr, narrow = (120 + phase) / dpr;
    const box: Box = { node: ROOT, rank: 0, count: 1, kind: "leaf", branch: 0,
      p: 0.5, q: 0.5, lnMass: Math.log(0.5), x0: 10 / 23, x1: right / 23,
      y0: 10 / view.height, y1: bottom / view.height };
    // Two children of the root, the second narrower, a node right of the
    // first, and a cousin below. Strokes are a box's own last column and
    // row, or the row above it: siblings meet unstroked but where the first
    // reaches past the second, and the last child's bottom is stroked.
    const sibling = { ...box, rank: 1, branch: 1, y0: box.y1, y1: 150 / dpr / view.height, x1: narrow / 23 };
    const frame = { boxes: [box, sibling,
      { ...box, node: 1n, x0: box.x1, x1: (220 / dpr) / 23 },
      { ...sibling, node: 2n, y0: sibling.y1, y1: 200 / dpr / view.height },
    ] } as Frame;
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    draw(ctx, frame, cam(), view, { p: 0, q: 0 }, palette, state);
    const pixels = ctx.getImageData(0, 0, c.width, c.height).data;
    const color = rgb(palette.tile(0.5, 0.5, true).fill);
    const stroke = rgb(palette.scheme.stroke);
    const difference = (x: number, y: number, expected: number[]) =>
      Math.max(...expected.map((v, i) => Math.abs(v - pixels[(y * c.width + x) * 4 + i]!)));
    // Edges as they round to physical pixels, within float error of a grid
    // line on it: their last column or row.
    const lastPixel = (css: number) => Math.ceil(css * dpr - 1e-6) - 1;
    const x = lastPixel(right), y = lastPixel(bottom), step = lastPixel(narrow);
    const last = lastPixel(sibling.y1 * view.height);
    const error = Math.max(
      difference(30, y, color), difference(step - 5, y, color), difference(step + 5, y, stroke),
      difference(step, y + 10, stroke), difference(step - 1, y + 10, color),
      difference(30, last, stroke), difference(30, last + 1, color),
      difference(x - 1, 30, color), difference(x, 30, stroke), difference(x + 1, 30, color));
    results.push({ name: "strokes outline each node's children, never between siblings", dpr, phase, error });
  }

  // A node's time branches are one CDF on the timeline drawn across the same
  // pixels: at the time each lands on, through a row between them,
  // and the time they move over, from where the path is, shaded.
  {
    const at = (rank: number, branch: number | null, y0: number, y1: number): Box => ({ node: ROOT, rank, count: 1,
      kind: branch === null ? "row" : "leaf", branch, p: 0.5, q: 0.5, lnMass: Math.log(0.5),
      x0: 10 / 23, x1: 200 / 23, y0: y0 / view.height, y1: y1 / view.height });
    // The first ends mid-pixel: drawn to the next grid line, and shaded there.
    const frame = { boxes: [at(0, 0, 10, 39.6), at(1, null, 40, 50), at(2, 2, 50, 80)] } as Frame;
    const timeline = { x: (time: number) => time, left: 0, right: view.width };
    const glyph = (_: bigint, branch: number) => ({ text: "", mark: "time" as const, time: { from: 20, to: [60, 100, 140][branch]! } });
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    // The row holds a known time branch too thin to draw, landing at 100.
    const known = () => [{ branch: 0, y0: 10 / view.height, y1: 40 / view.height },
      { branch: 1, y0: 44 / view.height, y1: 46 / view.height }, { branch: 2, y0: 50 / view.height, y1: 80 / view.height }];
    draw(ctx, frame, cam(), view, { p: 0, q: 0 }, palette, { ...state, glyph, timeline, known });
    const tile = palette.tile(0.5, 0.5, true);
    const fill = rgb(tile.fill), ink = rgb(tile.ink);
    const shaded = fill.map((v, i) => v + 0.28 * (ink[i]! - v));
    const pixel = (x: number, y: number) => ctx.getImageData(Math.floor(x * dpr), Math.floor(y * dpr), 1, 1).data;
    const difference = (x: number, y: number, expected: number[]) => Math.max(...expected.map((v, i) => Math.abs(v - pixel(x, y)[i]!)));
    // Within the row, the line stands at the known branch inside it.
    const error = Math.max(difference(40, 25, shaded), difference(40, 39.7, shaded), difference(60, 25, ink), difference(100, 25, fill),
      difference(50, 45, shaded), difference(100, 45, ink), difference(130, 45, fill), difference(140, 65, ink),
      difference(170, 65, fill));
    results.push({ name: "time branches are one CDF on the timeline", dpr, phase: 0, error });
  }

  // A child ending mid-pixel, then children thinner than a pixel in rows
  // after it: siblings meet unstroked, so a column through all of them
  // shows no stroke, however their edges fall on the grid.
  for (const phase of [0, 0.2, 0.61, 0.99]) {
    const camera = cam(phase);
    const lens = { p: 0, q: 0 };
    const frame = layout(midPixel, lens, camera, view, { rowPx: 1 / dpr });
    const ctx = canvas(dpr).getContext("2d")!;
    draw(ctx, frame, camera, view, lens, palette, state);
    const fill = rgb(palette.tile(0.5, 0.5, true).fill);
    const column = Math.floor(10 * dpr), top = Math.ceil((phase + 2) * dpr);
    const pixels = ctx.getImageData(column, top, 1, Math.floor(view.height * dpr) - top).data;
    // Short of the node's own last row, where its bottom stroke is.
    const end = Math.ceil(Math.max(...frame.boxes.map(b => b.y1)) * view.height * dpr - 1e-6) - 1;
    let error = 0;
    for (let i = 0; i < pixels.length && top + i / 4 < end; i += 4) {
      error = Math.max(error, ...fill.map((v, channel) => Math.abs(v - pixels[i + channel]!)));
    }
    results.push({ name: "rows after a child ending mid-pixel are never stroked over", dpr, phase, error });
    frame.dispose();
  }

  // A time branch shows the label its parent was too small for, though
  // never its own: its time is on the timeline, here off to the right.
  {
    const child = state.child(ROOT, 5);
    const leaf = (node: bigint, branch: number, x1: number, y0: number, y1: number): Box => ({ node, rank: 0, count: 1,
      kind: "leaf", branch, p: 0.5, q: 0.5, lnMass: Math.log(0.5), x0: 0, x1: x1 / 23, y0: y0 / view.height, y1: y1 / view.height });
    const frame = { boxes: [leaf(ROOT, 5, 3, 10, 200), { ...leaf(child, 0, 260, 10, 200), x0: 3 / 23 }] } as Frame;
    const timeline = { x: (time: number) => time, left: 0, right: view.width };
    const glyph = (node: bigint) => node === ROOT ? { text: "piano" } : { text: "9 s", mark: "time" as const, time: { from: 1000, to: 1000 } };
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    draw(ctx, frame, cam(), view, { p: 0, q: 0 }, palette, { ...state, glyph, timeline });
    // Carried labels are framed and faint: any pixel far from the tile.
    const fill = rgb(palette.tile(0.5, 0.5, true).fill);
    const pixels = ctx.getImageData(Math.ceil(10 * dpr), Math.ceil(20 * dpr), Math.floor(200 * dpr), Math.floor(160 * dpr)).data;
    let inked = 0;
    for (let i = 0; i < pixels.length; i += 4) inked += Number(60 < Math.max(...fill.map((v, k) => Math.abs(v - pixels[i + k]!))));
    results.push({ name: "a time branch shows what it carries", dpr, phase: 0, error: inked === 0 ? 255 : 0 });
  }

  // Thin children between a bordered one and the background: each physical
  // row must show the box the pointer over it selects.
  for (const phase of [0, 0.2, 0.61, 0.99]) {
    const camera = { ...cam(), anchor: { path: [], digits: digits(0.99) }, a: 0.5 + phase / dpr / view.height };
    const frame = layout(thin, comparison, camera, view, { rowPx: 1 / dpr });
    const ctx = canvas(dpr).getContext("2d")!;
    draw(ctx, frame, camera, view, comparison, palette, state);
    const column = Math.floor(10 * dpr);
    const top = Math.floor(view.height * dpr / 2) - 3;
    const pixels = ctx.getImageData(column, top, 1, 12).data;
    let error = 0;
    for (let row = 0; row < 12; row++) {
      const y = (top + row) / dpr;
      const box = pick(thin, comparison, frame, camera, view, column / dpr, y, 1 / dpr)!.box;
      const drawn = frame.boxes.find(b => b.node === box.node && b.branch === box.branch);
      // The outline of the node's children: the row above the first, and the
      // last's own last row, or the row below it where it is one row tall.
      const group = frame.boxes.filter(b => b.node === ROOT);
      const rowOf = (y: number) => Math.ceil(y * view.height * dpr - 1e-6);
      const lowest = group.reduce((a, b) => (a.y1 < b.y1 ? b : a), group[0]!);
      const strokes = [rowOf(Math.min(...group.map(b => b.y0))) - 1,
        rowOf(lowest.y1) - rowOf(lowest.y0) <= 1 ? rowOf(lowest.y1) : rowOf(lowest.y1) - 1];
      const separator = strokes.includes(top + row);
      const expected = rgb(separator ? palette.scheme.stroke : drawn === undefined ? palette.scheme.surface
        : palette.tile(box.p, box.q, false).fill);
      for (let channel = 0; channel < 3; channel++) {
        error = Math.max(error, Math.abs(expected[channel]! - pixels[row * 4 + channel]!));
      }
    }
    results.push({ name: "pointer selects what its pixel shows", dpr, phase, error });
    frame.dispose();
  }

  // Rows of known children between gaps: every row selects a branch, and
  // its highlight covers the row the pointer is on.
  for (const phase of [0, 0.2, 0.61]) {
    const camera = cam(phase);
    const lens = { p: 0, q: 0 };
    const frame = layout(gapped, lens, camera, view, { rowPx: 1 / dpr });
    const column = Math.floor(10 * dpr);
    const ink = rgb(palette.scheme.hover);
    let error = 0;
    for (let row = 20; row < 60; row++) {
      const hover = pick(gapped, lens, frame, camera, view, column / dpr, row / dpr, 1 / dpr)!;
      const ctx = canvas(dpr).getContext("2d")!;
      draw(ctx, frame, camera, view, lens, palette, { ...state, hover });
      const pixel = ctx.getImageData(column, row, 1, 1).data;
      const missing = Math.max(...ink.map((v, i) => Math.abs(v - pixel[i]!)));
      error = Math.max(error, hover.box.branch === null ? 255 : missing);
    }
    results.push({ name: "a row's pixel selects and highlights a branch", dpr, phase, error });
    frame.dispose();
  }
}
const failed = results.filter(r => 1 < r.error);
document.getElementById("result")!.textContent = `${failed.length ? "FAIL" : "PASS"}: ${results.length} canvas pixel checks\n`
  + JSON.stringify(failed.length ? failed : results, null, 2);
await fetch("/result", { method: "POST", body: JSON.stringify({ passed: failed.length === 0, results }) });
