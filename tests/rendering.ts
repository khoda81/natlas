// Real Canvas pixel checks, served by scripts/check-rendering.ts.
import { layout, linearPalette, pick, top, type Frame } from "../src/core/layout";
import { IntervalTree, type Lens, ROOT } from "../src/core/tree";
import { TextLabels } from "../src/readings/labels";
import { Palette } from "../src/ui/color";
import { draw, NATS, RULER_HEIGHT } from "../src/ui/render";
import { answer, digits, run } from "./build";

const view = { width: 270, height: 340 };
const palette = Palette.current();
const state = { hover: null, units: NATS, glyph: () => ({ text: "" }), note: null, timeline: null, known: () => [] };
const cam = (phase = 0, k = 23) => ({ origin: ROOT, x: 0, k, anchor: top(), a: phase / view.height });
const rgb = (color: string) => color.slice(1).match(/../g)!.map(c => Number.parseInt(c, 16));
const canvas = (dpr: number) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(view.width * dpr);
  canvas.height = Math.round((view.height + RULER_HEIGHT) * dpr);
  canvas.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);
  return canvas;
};
const [fill, stroke, background, hover, ink] = [palette.scheme.tile, palette.scheme.stroke, palette.scheme.surface,
  palette.scheme.hover, palette.ink].map(rgb) as [number[], number[], number[], number[], number[]];
const differs = (pixels: Uint8ClampedArray, at: number, expected: number[]) =>
  Math.max(...expected.map((v, channel) => Math.abs(v - pixels[at + channel]!)));

/** Every branch of a distribution, given once and ending the node. */
const complete = (weights: number[], branches?: number[]) => answer(weights, branches, 0, 0);
const count = 10000;
const uniform = new IntervalTree();
uniform.insert(ROOT, 0, complete(Array(count).fill(1 / count)));
const varying = new IntervalTree();
const weights = Array.from({ length: count }, (_, i) => 1 + 0.9 * Math.cos(i / (count - 1) * Math.PI));
const total = weights.reduce((a, b) => a + b, 0);
const p = weights.map(w => w / total);
varying.insert(ROOT, 0, complete(p));
varying.insert(ROOT, 1, complete(p, p.map((_, i) => count - 1 - i)));
// Thick children of distinct widths, and steps between them.
const steps = new IntervalTree();
steps.insert(ROOT, 0, complete([0.5, 0.3, 0.15, 0.05]));
const boundaries = new IntervalTree();
boundaries.insert(ROOT, 0, complete([0.5, 0.5]));
boundaries.insert(ROOT, 1, complete([Math.exp(-50), Math.exp(-50)]));
const aaColors = linearPalette(Number.parseInt(palette.scheme.surface.slice(1), 16), Number.parseInt(palette.scheme.stroke.slice(1), 16));
const halfCoverage = rgb(aaColors[aaColors.length / 2 - 1]!);
// A heavy child ending mid-pixel at every ratio, then children thinner than
// a pixel.
const midPixel = new IntervalTree();
midPixel.insert(ROOT, 0, complete([0.5, 0.4983, 0.0007, 0.0006, 0.0004]));
// Known children alternate with one-child gaps, so the tail spans runs.
const gapped = new IntervalTree();
gapped.insert(ROOT, 0, { runs: Array.from({ length: count / 2 }, (_, i) =>
  run([1 / count], [2 * i], 2 * i, 1 - (2 * i + 1) / count)) });

/**
 * What every physical pixel of the root's children shows, by the rule
 * itself: one any edge of them falls in, each pixel the half-open square
 * from its corner, is outline; one whose middle is inside them is filled;
 * any other is background. Their edges: where they start, the node's top
 * and bottom, each child's end, and the shared horizontal edges of two children.
 */
function oracle(tree: IntervalTree, lens: Lens, frame: Frame, camera: ReturnType<typeof cam>, dpr: number): Uint8Array {
  const [columns, rows] = [Math.round(view.width * dpr), Math.floor(view.height * dpr)];
  const shown = new Uint8Array(columns * rows); // 0 background, 1 fill, 2 stroke
  const X = (nats: number) => (nats - camera.x) * camera.k * dpr;
  const Y = (y: number) => y * view.height * dpr;
  const cell = (at: number) => Math.floor(at + 1e-6);
  const children = frame.known(tree, lens, ROOT).map(({ branch, y0, y1 }) =>
    ({ y0: Y(y0), y1: Y(y1), x1: X(tree.entry(ROOT, lens, branch)!.natsQ) }));
  const x0 = X(0);
  for (const child of children) {
    for (let r = Math.max(0, Math.floor(child.y0)); r < Math.min(rows, Math.ceil(child.y1)); r++) {
      if (!(child.y0 <= r + 0.5 && r + 0.5 < child.y1)) continue;
      for (let c = Math.max(0, Math.floor(x0)); c < Math.min(columns, child.x1 - 0.5); c++) {
        if (x0 <= c + 0.5) shown[r * columns + c] = 1;
      }
    }
  }
  const mark = (r0: number, r1: number, c0: number, c1: number) => {
    for (let r = Math.max(0, r0); r <= Math.min(rows - 1, r1); r++) {
      for (let c = Math.max(0, c0); c <= Math.min(columns - 1, c1); c++) shown[r * columns + c] = 2;
    }
  };
  const [first, last] = [children[0]!, children.at(-1)!];
  mark(cell(first.y0), cell(last.y1 - 2e-6), cell(x0), cell(x0));
  mark(cell(first.y0), cell(first.y0), cell(x0), cell(first.x1));
  mark(cell(last.y1), cell(last.y1), cell(x0), cell(last.x1));
  children.forEach((child, i) => {
    mark(cell(child.y0), cell(child.y1 - 2e-6), cell(child.x1), cell(child.x1));
    const next = children[i + 1];
    if (next !== undefined) mark(cell(child.y1), cell(child.y1), cell(x0), cell(Math.max(child.x1, next.x1)));
  });
  return shown;
}

const results: { name: string; dpr: number; phase: number; error: number }[] = [];
for (const dpr of [1, 1.25, 1.5, 2]) {
  const halfRow = Math.floor(view.height * dpr / 2);
  const gridPhase = halfRow / dpr - view.height / 2;
  for (const [name, camera, lens, row, color] of [
    ["half-covered boundary row", cam(0.5 / dpr), { p: 0, q: 0 }, 0, halfCoverage],
    ["grid-aligned sibling boundary", cam(gridPhase), { p: 0, q: 0 }, halfRow, stroke],
    ["deep exact sibling boundary", { ...cam(), x: 45 * Math.LN2, a: halfRow / (view.height * dpr), anchor: top([1]) }, { p: 0, q: 1 }, halfRow, stroke],
  ] as const) {
    const frame = layout(boundaries, lens, camera, view, { rowPx: 1 / dpr });
    const ctx = canvas(dpr).getContext("2d")!;
    draw(ctx, frame, camera, view, palette, { ...state, raster: { tree: boundaries, lens } });
    const pixel = ctx.getImageData(2, row, 1, 1).data;
    results.push({ name, dpr, phase: camera.a * view.height, error: Math.max(differs(pixel, 0, color), 255 - pixel[3]!) });
    frame.dispose();
  }

  for (const [name, tree, lens] of [
    ["steps", steps, { p: 0, q: 0 }],
    ["uniform tail", uniform, { p: 0, q: 0 }],
    ["comparison widths", varying, { p: 0, q: 1 }],
    ["a child ending mid-pixel, then thin ones", midPixel, { p: 0, q: 0 }],
  ] as const) {
    for (const phase of [0, 0.2, 0.61, 0.99]) {
      const camera = cam(phase);
      const frame = layout(tree, lens, camera, view, { rowPx: 1 / dpr });
      const c = canvas(dpr), ctx = c.getContext("2d")!;
      draw(ctx, frame, camera, view, palette, state);
      const expected = oracle(tree, lens, frame, camera, dpr);
      const pixels = ctx.getImageData(0, 0, c.width, Math.floor(view.height * dpr)).data;
      let error = 0;
      for (let i = 0; i < expected.length; i++) {
        error = Math.max(error, differs(pixels, 4 * i, [background, fill, stroke][expected[i]!]!));
      }
      results.push({ name: `pixels by containment: ${name}`, dpr, phase, error });
      frame.dispose();
    }
  }

  // Branch fills follow the resulting role, including on its control token.
  // The same sibling boundary pixels are still drawn over every fill.
  {
    const tree = new IntervalTree();
    const tokens = ["<|im_start|>", "user\n", "assistant\n", "<think>", "</think>", "<|im_end|>", " text"];
    const utf8 = new TextEncoder();
    const { conversation } = new TextLabels(tokens.map(token => utf8.encode(token))).read(tree);
    const { user, assistant, thinking } = palette.scheme.conversation;
    let error = 0;
    for (const [path, branches, colors] of [
      [[0], [1, 2], [user, assistant]],
      [[0, 2], [6, 3, 5, 0], [assistant, thinking, palette.scheme.tile, palette.scheme.tile]],
      [[0, 2, 3], [6, 4], [thinking, assistant]],
    ] as const) {
      const origin = tree.descend(ROOT, path);
      tree.insert(origin, 0, complete(branches.map(() => 1 / branches.length), [...branches]));
      const camera = { ...cam(0, 100), origin };
      const frame = layout(tree, { p: 0, q: 0 }, camera, view, { rowPx: 1 / dpr });
      const c = canvas(dpr), ctx = c.getContext("2d")!;
      draw(ctx, frame, camera, view, palette, { ...state, conversation });
      for (const [i, color] of colors.entries()) {
        const height = view.height * dpr / branches.length;
        const column = Math.floor(12 * dpr);
        const interior = ctx.getImageData(column, Math.floor((i + 0.5) * height), 1, 1).data;
        const boundary = ctx.getImageData(column, Math.floor(i * height + 1e-6), 1, 1).data;
        error = Math.max(error, differs(interior, 0, rgb(color)), differs(boundary, 0, stroke));
      }
      frame.dispose();
    }
    results.push({ name: "ChatML branches tint the fill and retain their outlines", dpr, phase: 0, error });
  }

  // Three boundaries of equal-width siblings occupy one row when small.
  // After zoom they are separate lines, including after the camera pans.
  {
    const tree = new IntervalTree();
    const cuts = [0.5, 0.5004, 0.5008];
    tree.insert(ROOT, 0, complete([cuts[0]!, cuts[1]! - cuts[0]!, cuts[2]! - cuts[1]!, 1 - cuts[2]!]));
    tree.insert(ROOT, 1, answer(Array(4).fill(Math.exp(-7))));
    const lens = { p: 0, q: 1 };
    const anchor = { path: [], digits: digits(0.5) };
    let error = 0;
    for (const [x, pan] of [[0, 0.1], [4, 0.1], [4, 0.77]] as const) {
      const a = (Math.floor(0.4 * view.height * dpr) + pan) / (view.height * dpr);
      const camera = { ...cam(0, 40), x, anchor, a };
      const frame = layout(tree, lens, camera, view, { rowPx: 1 / dpr });
      const ctx = canvas(dpr).getContext("2d")!;
      draw(ctx, frame, camera, view, palette, state);
      const rows = cuts.map(cut => Math.floor((a + (cut - 0.5) * Math.exp(x)) * view.height * dpr + 1e-6));
      const column = Math.floor(30 * dpr);
      const color = (row: number, rgb: number[]) => differs(ctx.getImageData(column, row, 1, 1).data, 0, rgb);
      for (const row of rows) error = Math.max(error, color(row, stroke));
      if (x === 0) {
        if (new Set(rows).size !== 1) error = 255;
      } else {
        if (new Set(rows).size !== cuts.length) error = 255;
        for (const row of rows.slice(0, -1)) error = Math.max(error, color(row + 1, fill));
      }
      frame.dispose();
    }
    results.push({ name: "sibling boundaries stay visible and split on zoom", dpr, phase: 0, error });
  }

  // A node's time branches are one CDF on the timeline drawn across the same
  // pixels: at the time each lands on, even one too thin to draw, and the
  // time they move over, from where the path is, shaded.
  {
    const tree = new IntervalTree();
    // 136 pixels, a fraction of one, and the rest; wider than the view, but
    // the last, 153 pixels.
    tree.insert(ROOT, 0, complete([0.4, 0.0004, 0.5996]));
    const camera = cam(0, 300);
    const frame = layout(tree, { p: 0, q: 0 }, camera, view, { rowPx: 1 / dpr });
    const timeline = { x: (time: number) => time, left: 0, right: view.width };
    // Time stood at 12, then 24, before 30: shaded from 12 on.
    const trail = { at: 24, shaded: true, before: { at: 12, shaded: true, before: null } };
    const glyph = (_: bigint, branch: number) => ({ text: "", mark: "time" as const, time: { from: 30, to: [60, 100, 140][branch]!, shaded: true, trail } });
    const known = () => frame.known(tree, { p: 0, q: 0 }, ROOT);
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    draw(ctx, frame, camera, view, palette, { ...state, glyph, timeline, known });
    // Shaded time is ink over the window's own base, which the rest of it shows.
    const base = rgb(palette.scheme.window);
    const shaded = base.map((v, i) => v + 0.16 * (ink[i]! - v));
    const pixels = ctx.getImageData(0, 0, c.width, c.height).data;
    const at = (x: number, y: number) => 4 * (Math.floor(y * dpr) * c.width + Math.floor(x * dpr));
    const error = Math.max(differs(pixels, at(40, 70), shaded), differs(pixels, at(60, 70), ink),
      differs(pixels, at(100, 70), base), differs(pixels, at(120, 250), shaded), differs(pixels, at(140, 250), ink),
      differs(pixels, at(148, 250), base),
      // Behind 30 the intervals back to 24 and to 12 shaded as the trail says, and before it nothing.
      differs(pixels, at(27, 70), shaded), differs(pixels, at(18, 250), shaded), differs(pixels, at(8, 70), base));
    results.push({ name: "time branches are one CDF on the timeline", dpr, phase: 0, error });
    frame.dispose();
  }

  // A time branch shows the label its parent was too small for, though
  // never its own: its time is on the timeline, here off to the right.
  {
    const tree = new IntervalTree();
    tree.insert(ROOT, 0, complete([0.98, 0.02]));
    tree.insert(tree.child(ROOT, 0), 0, complete([0.5, 0.5]));
    const camera = cam(0, 300);
    const frame = layout(tree, { p: 0, q: 0 }, camera, view, { rowPx: 1 / dpr });
    const timeline = { x: (time: number) => time, left: 0, right: view.width };
    const glyph = (node: bigint) => node === ROOT ? { text: "piano" } : { text: "9 s", mark: "time" as const, time: { from: 1000, to: 1000, shaded: true, trail: null } };
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    draw(ctx, frame, camera, view, palette, { ...state, glyph, timeline });
    // Carried labels are framed and faint: any pixel inside the children
    // far from both the fill and the outline.
    const pixels = ctx.getImageData(Math.ceil(20 * dpr), Math.ceil(20 * dpr), Math.floor(200 * dpr), Math.floor(120 * dpr)).data;
    let inked = 0;
    for (let i = 0; i < pixels.length; i += 4) inked += Number(60 < differs(pixels, i, fill) && 60 < differs(pixels, i, stroke));
    results.push({ name: "a time branch shows what it carries", dpr, phase: 0, error: inked === 0 ? 255 : 0 });
    frame.dispose();
  }

  // Known children between gaps, forty a pixel: every row selects a branch,
  // and its highlight covers the row the pointer is on.
  for (const phase of [0, 0.2, 0.61]) {
    const camera = cam(phase);
    const lens = { p: 0, q: 0 };
    const frame = layout(gapped, lens, camera, view, { rowPx: 1 / dpr });
    const column = Math.floor(10 * dpr);
    let error = 0;
    for (let row = 20; row < 60; row++) {
      const selected = pick(gapped, lens, frame, camera, view, column / dpr, row / dpr, 1 / dpr)!;
      const ctx = canvas(dpr).getContext("2d")!;
      draw(ctx, frame, camera, view, palette, { ...state, hover: selected });
      const pixel = ctx.getImageData(column, row, 1, 1).data;
      error = Math.max(error, selected.box.branch === null ? 255 : differs(pixel, 0, hover));
    }
    results.push({ name: "a pixel of thin children selects and highlights a branch", dpr, phase, error });
    frame.dispose();
  }
}
const failed = results.filter(r => 1 < r.error);
document.getElementById("result")!.textContent = `${failed.length ? "FAIL" : "PASS"}: ${results.length} canvas pixel checks\n`
  + JSON.stringify(failed.length ? failed : results, null, 2);
await fetch("/result", { method: "POST", body: JSON.stringify({ passed: failed.length === 0, results }) });
