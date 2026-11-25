import type { Camera, Viewport } from "./layout";

// One explicit initialization for both the browser bundle and Bun's test runner.
import init, {
  aim,
  clampPosition,
  extent,
  layout as wasmLayout,
  LayoutFrame,
  project as wasmProject,
  RecordedTree,
  reach,
  Session,
  settle,
  Tree,
  view,
} from "./generated/natlas_core";

const wasm = typeof document === "undefined"
  ? new URL("./generated/natlas_core_bg.wasm", import.meta.url)
  : new URL("generated/natlas_core_bg.wasm", document.baseURI);
await init({ module_or_path: wasm });

const emptyPath = new Uint32Array();
function frame(
  run: typeof wasmLayout | typeof wasmProject,
  tree: Tree,
  p: number,
  q: number,
  { origin, x, k, a, anchor }: Camera,
  { width, height }: Viewport,
  pixels: number,
): LayoutFrame {
  const path = anchor.path.length === 0 ? emptyPath : Uint32Array.from(anchor.path);
  return run(tree, p, q, origin, x, k, a, path, anchor.digits, width, height, pixels);
}

export function layout(tree: Tree, p: number, q: number, camera: Camera, view: Viewport, rowPx: number): LayoutFrame {
  return frame(wasmLayout, tree, p, q, camera, view, rowPx);
}

export function project(tree: Tree, p: number, q: number, camera: Camera, view: Viewport, pixelRatio: number): LayoutFrame {
  return frame(wasmProject, tree, p, q, camera, view, pixelRatio);
}

export {
  aim,
  clampPosition,
  extent,
  LayoutFrame,
  RecordedTree,
  reach,
  Session,
  settle,
  Tree,
  view,
};
