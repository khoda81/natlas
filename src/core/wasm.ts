// One explicit initialization for both the browser bundle and Bun's test runner.
import init, {
  clampPosition,
  decodeListing,
  encodeListing,
  layout,
  normalizeBoundary,
  pick,
  place,
  reanchor,
  repairBoundary,
  rootExtent,
  Tree,
} from "./generated/natlas_core";

const wasm = typeof document === "undefined"
  ? new URL("./generated/natlas_core_bg.wasm", import.meta.url)
  : new URL("generated/natlas_core_bg.wasm", document.baseURI);
await init({ module_or_path: wasm });

export {
  clampPosition,
  decodeListing,
  encodeListing,
  layout,
  normalizeBoundary,
  pick,
  place,
  reanchor,
  repairBoundary,
  rootExtent,
  Tree,
};
