// One explicit initialization for both the browser bundle and Bun's test runner.
import init, {
  aim,
  clampPosition,
  extent,
  layout,
  LayoutFrame,
  Session,
  settle,
  Tree,
  view,
} from "./generated/natlas_core";

const wasm = typeof document === "undefined"
  ? new URL("./generated/natlas_core_bg.wasm", import.meta.url)
  : new URL("generated/natlas_core_bg.wasm", document.baseURI);
await init({ module_or_path: wasm });

export {
  aim,
  clampPosition,
  extent,
  layout,
  LayoutFrame,
  Session,
  settle,
  Tree,
  view,
};
