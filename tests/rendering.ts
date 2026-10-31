// Real Canvas pixel checks, served by scripts/check-rendering.ts.
import { layout, type Box, type Frame } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { Palette, rampPosition } from "../src/ui/color";
import { draw, NATS, RULER_HEIGHT } from "../src/ui/render";
import { answer, run } from "./build";

const view = { width: 270, height: 340 };
const palette = Palette.current();
const state = { hover: null, ancestors: new Set<string>(), units: NATS, glyph: () => ({ text: "" }), note: null };
const cam = (phase = 0) => ({ origin: ROOT, x: 0, k: 23, anchor: { node: ROOT, rank: 0 }, a: phase / view.height });
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
      const frame = layout(tree, lens, camera, view, { mergePx: 1 / dpr });
      const ctx = canvas(dpr).getContext("2d")!;
      draw(ctx, frame, camera, view, lens, palette, state);
      const pixels = ctx.getImageData(Math.round(10 * dpr), 0, 1, Math.floor(view.height * dpr)).data;
      const ramp = lens.p === lens.q ? palette.sequential : palette.diverging;
      let index = 0, error = 0;
      for (let row = 2; row < view.height * dpr - 2; row++) {
        const top = row / dpr / view.height, bottom = (row + 1) / dpr / view.height;
        while (index + 1 < frame.boxes.length && frame.boxes[index]!.y1 <= top) index++;
        const colors: number[][] = [];
        for (let i = index; i < frame.boxes.length && frame.boxes[i]!.y0 < bottom; i++) {
          const box = frame.boxes[i]!;
          colors.push(rgb(ramp.fill[ramp.level(rampPosition(box.p, box.q, lens.p === lens.q))]!));
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
    const right = (160 + phase) / dpr, bottom = (80 + phase) / dpr;
    const box: Box = { node: ROOT, rank: 0, count: 1, kind: "leaf", branch: 0,
      p: 0.5, q: 0.5, lnMass: Math.log(0.5), x0: 10 / 23, x1: right / 23,
      y0: 10 / view.height, y1: bottom / view.height };
    // Neighbours below and to the right would cover a prematurely painted
    // border. Their interior pixels must remain fully filled either side.
    const frame = { boxes: [box,
      { ...box, rank: 1, branch: 1, y0: box.y1, y1: 150 / dpr / view.height },
      { ...box, node: 1n, x0: box.x1, x1: (220 / dpr) / 23 },
    ] } as Frame;
    const c = canvas(dpr), ctx = c.getContext("2d")!;
    draw(ctx, frame, cam(), view, { p: 0, q: 0 }, palette, state);
    const pixels = ctx.getImageData(0, 0, c.width, c.height).data;
    const color = rgb(palette.sequential.fill[palette.sequential.level(rampPosition(0.5, 0.5, true))]!);
    const background = rgb(palette.scheme.surface);
    const difference = (x: number, y: number, expected: number[]) =>
      Math.max(...expected.map((v, i) => Math.abs(v - pixels[(y * c.width + x) * 4 + i]!)));
    const x = Math.floor(right * dpr), y = Math.floor(bottom * dpr);
    const error = Math.max(difference(30, y - 1, color), difference(30, y, background), difference(30, y + 1, color),
      difference(x - 1, 30, color), difference(x, 30, background), difference(x + 1, 30, color));
    results.push({ name: "boundary owns one physical pixel", dpr, phase, error });
  }
}
const failed = results.filter(r => 1 < r.error);
document.getElementById("result")!.textContent = `${failed.length ? "FAIL" : "PASS"}: ${results.length} canvas pixel checks\n`
  + JSON.stringify(failed.length ? failed : results, null, 2);
await fetch("/result", { method: "POST", body: JSON.stringify({ passed: failed.length === 0, results }) });
