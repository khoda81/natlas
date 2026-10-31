// Run after `bun run core`, then open the printed URL in the target browser.
const built = await Bun.build({ entrypoints: ["tests/rendering.ts"], target: "browser" });
if (!built.success) throw new Error(String(built.logs));
const out = process.env.NATLAS_RENDER_OUTPUT ?? "/tmp/natlas-rendering-tests.json";
const server = Bun.serve({ hostname: "127.0.0.1", port: Number(process.env.PORT ?? 5191),
  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === "/result" && request.method === "POST") {
      await Bun.write(out, await request.text());
      return new Response("saved");
    }
    if (path === "/checks.js") return new Response(built.outputs[0], { headers: { "content-type": "text/javascript" } });
    if (path === "/generated/natlas_core_bg.wasm") return new Response(Bun.file("src/core/generated/natlas_core_bg.wasm"),
      { headers: { "content-type": "application/wasm" } });
    if (path === "/") return new Response(`<!doctype html><html data-theme="dark"><meta charset="utf-8">
      <title>Natlas rendering checks</title><pre id="result">Checking canvas pixels…</pre>
      <script type="module" src="/checks.js"></script></html>`, { headers: { "content-type": "text/html" } });
    return new Response("not found", { status: 404 });
  },
});
console.log(`Rendering checks: ${server.url} (report: ${out})`);
