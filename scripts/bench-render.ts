// Run `bun scripts/bench-render.ts`, open the printed URL, and click Run.
import { resolve } from "node:path";

const out = resolve(process.env.NATLAS_PERF_OUTPUT ?? "/tmp/natlas-render-benchmark");
const wasm = Bun.file(new URL("../src/core/generated/natlas_core_bg.wasm", import.meta.url));
const html = `<!doctype html>
<html>
<head><meta charset="utf-8"><title>natlas render & transfer benchmark</title></head>
<body style="margin:0;font-family:monospace;background:#111;color:#eee">
  <canvas id="canvas" style="display:block;border-bottom:1px solid #333"></canvas>
  <aside style="padding:12px">
    <button id="benchmark-run" style="padding:6px 12px;font-size:14px" disabled>Run render &amp; transfer benchmark</button>
    <pre id="benchmark-output" style="margin-top:10px;white-space:pre-wrap">Loading WebAssembly…</pre>
  </aside>
  <script type="module" src="/benchmark.js"></script>
</body>
</html>`;

const built = await Bun.build({
  entrypoints: ["tests/render-benchmark.ts"],
  target: "browser",
  minify: true,
});
if (!built.success) throw new Error(String(built.logs));
const bundle = await built.outputs[0]!.text();
const headers = {
  "cross-origin-opener-policy": "same-origin",
  "cross-origin-embedder-policy": "require-corp",
};

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: Number(process.env.PORT ?? 5188),
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/result" && request.method === "POST") {
      if (request.headers.get("content-type") !== "application/json") {
        return new Response("expected a JSON report", { status: 415 });
      }
      const result = await request.json().catch(() => null);
      if (typeof result?.label !== "string" || !/^[a-z0-9-]+$/.test(result.label)) {
        return new Response("invalid label", { status: 400 });
      }
      await Bun.write(`${out}/${result.label}.json`, JSON.stringify(result, null, 2));
      return new Response("saved", { headers });
    }
    if (url.pathname === "/") return new Response(html, { headers: { ...headers, "content-type": "text/html" } });
    if (url.pathname === "/benchmark.js") return new Response(bundle, { headers: { ...headers, "content-type": "text/javascript" } });
    if (url.pathname === "/generated/natlas_core_bg.wasm") {
      return new Response(wasm, {
        headers: { ...headers, "content-type": "application/wasm" },
      });
    }
    return new Response("not found", { status: 404 });
  },
});

console.log(`Render & transfer benchmark: ${server.url} (reports: ${out})`);
